"""
================================================================
NEUMA Phase 8 — RD-GANet Training Loop  (multi-GPU edition)
================================================================
Handles:
  - DataParallel wrapping (automatic when NUM_GPUS > 1)
  - Mixed-precision training via torch.cuda.amp (USE_AMP)
  - Gradient accumulation (GRAD_ACCUM_STEPS)
  - Single-fold training with early stopping
  - DP-aware checkpoint save / load  (model.module.state_dict)
  - Per-epoch performance logging:
      epoch duration, GPU memory, throughput (samples/s)
  - MMD domain-adaptation (source/target split from batch)

DataParallel notes
------------------
  • model.module is the raw network; always use it for compute_loss,
    checkpointing, and explainability hooks.
  • DataParallel scatters the batch across GPU replicas, runs forward,
    and gathers outputs on DEVICE (cuda:0).  Backward + optimizer run
    once on the gathered gradients.
  • GradScaler.scale() is applied *before* .backward(); it is safe
    with DataParallel as long as all forward paths use autocast.

OOM prevention
--------------
  • Gradient accumulation: effective_batch = BATCH_SIZE × GRAD_ACCUM_STEPS.
    Adjust GRAD_ACCUM_STEPS to increase effective batch without memory cost.
  • AMP (fp16 activations) roughly halves activation memory.
  • torch.cuda.empty_cache() is called between folds when Trainer is used
    inside LOSOCV (the caller's responsibility).
================================================================
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from .metrics import MetricTracker
from .losses  import MultiTaskLoss, temporal_consistency_loss

try:
    from config.settings import USE_TEMPORAL_LOSS, TEMPORAL_LOSS_WEIGHT
except ImportError:
    USE_TEMPORAL_LOSS    = False
    TEMPORAL_LOSS_WEIGHT = 0.10


class Trainer:
    """
    Trains one RDGANet model for one LOSOCV fold.

    Parameters
    ----------
    model          : RDGANet instance (raw, not yet DataParallel-wrapped)
    device         : primary torch.device
    loss_fn        : MultiTaskLoss
    lr             : learning rate
    weight_decay   : AdamW weight decay
    patience       : early-stopping patience (epochs without improvement)
    ckpt_path      : checkpoint save path
    use_dp         : wrap with DataParallel if NUM_GPUS > 1  (default True)
    use_amp        : mixed-precision training  (default: settings.USE_AMP)
    grad_accum_steps: micro-batches per optimiser step  (default: settings.GRAD_ACCUM_STEPS)

    Attributes (set after fit())
    ----------------------------
    perf_log       : list of per-epoch dicts
                     {epoch, duration_s, throughput_sps, gpu_alloc_gb,
                      peak_mem_gb, cumulative_s}
    best_val_acc   : best validation accuracy seen during training
    """

    def __init__(
        self,
        model         ,
        device        : torch.device,
        loss_fn       : MultiTaskLoss,
        lr            : float = 1e-3,
        weight_decay  : float = 1e-4,
        patience      : int   = 20,
        ckpt_path     : Path  = Path("ckpt.pt"),
        use_dp        : bool  = True,
        use_amp       : Optional[bool] = None,
        grad_accum_steps: Optional[int] = None,
    ):
        self.device    = device
        self.loss_fn   = loss_fn
        self.patience  = patience
        self.ckpt_path = Path(ckpt_path)
        self.perf_log  : list = []
        self.best_val_acc = -1.0
        self.best_val_f1  = -1.0
        self.temperature  = 1.0   # updated by calibrate_temperature()

        # ── DataParallel wrapping ────────────────────────────────────────────
        n_gpus = torch.cuda.device_count()
        if use_dp and n_gpus > 1 and device.type == "cuda":
            self._is_dp = True
            device_ids  = list(range(n_gpus))
            print(f"[INFO] Trainer: wrapping model in DataParallel "
                  f"({n_gpus} GPUs — device_ids={device_ids})")
            self.model = nn.DataParallel(model, device_ids=device_ids)
            self.model = self.model.to(device)
        else:
            self._is_dp = False
            self.model = model.to(device)

        # ── AMP ──────────────────────────────────────────────────────────────
        if use_amp is None:
            try:
                from config.settings import USE_AMP as _cfg_amp
                use_amp = _cfg_amp and device.type == "cuda"
            except ImportError:
                use_amp = device.type == "cuda"
        self.use_amp     = bool(use_amp)
        self._amp_device = device.type   # "cuda" or "cpu"
        self.scaler      = GradScaler("cuda") if self.use_amp else None

        # ── Gradient accumulation ────────────────────────────────────────────
        if grad_accum_steps is None:
            try:
                from config.settings import GRAD_ACCUM_STEPS as _ga
                grad_accum_steps = _ga
            except ImportError:
                grad_accum_steps = 1
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # ── Optimiser / scheduler ────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # CosineAnnealingWarmRestarts: restarts every T_0 epochs so LR never
        # collapses to zero.  Warm restarts help escape local minima that
        # ReduceLROnPlateau tends to get stuck in after its first LR halving.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0     = 50,
            T_mult  = 1,
            eta_min = 1e-6,
        )

    # ── Model helpers ────────────────────────────────────────────────────────

    def _raw_model(self) -> nn.Module:
        """Return the unwrapped model (strips DataParallel if present)."""
        return self.model.module if self._is_dp else self.model

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, best_val_bal: float, no_improve: int) -> None:
        """Atomically write full training state to self.ckpt_path.

        Writes to a .tmp sibling first, then renames so a concurrent reader
        never sees a partially-written file (POSIX rename is atomic on the
        same filesystem).
        """
        state = {
            "model_state_dict":     self._raw_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict() if self.scaler else None,
            "epoch":                epoch,
            "best_val_bal":         best_val_bal,
            "no_improve":           no_improve,
        }
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        try:
            tmp.rename(self.ckpt_path)
        except OSError:
            import shutil
            shutil.copy2(tmp, self.ckpt_path)
            tmp.unlink(missing_ok=True)

    def _load_checkpoint(self, path: Path) -> Optional[dict]:
        """Load a checkpoint and restore all available states.

        Handles two formats:
          • Full-state  (new)  — dict with "model_state_dict" key.
          • Weights-only (old) — bare OrderedDict of model parameters.

        Returns a resume dict {epoch, best_val_bal, no_improve} for full-state
        checkpoints, or None for legacy checkpoints (caller starts from epoch 1).
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except Exception:
            ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self._raw_model().load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            return {
                "epoch":        ckpt["epoch"],
                "best_val_bal": ckpt["best_val_bal"],
                "no_improve":   ckpt["no_improve"],
            }

        # Legacy weights-only checkpoint — restore model weights only.
        self._raw_model().load_state_dict(ckpt)
        return None

    # ── Batch utilities ──────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _split_mmd_embs(
        self, embs: torch.Tensor, subj_ids: torch.Tensor
    ):
        """Split embeddings by subject for inter-subject MMD.

        MMD requires population-level statistics; a single embedding per
        subject (typical when batch_size << n_subjects) gives a degenerate
        kernel estimate (variance=0).  Require at least MIN_MMD_PER_SIDE
        samples on each side before computing MMD.
        """
        MIN_MMD_PER_SIDE = 8
        unique = subj_ids.unique()
        if len(unique) < 2:
            return None, None
        s_mask = subj_ids == unique[0]
        src, tgt = embs[s_mask], embs[~s_mask]
        if src.shape[0] < MIN_MMD_PER_SIDE or tgt.shape[0] < MIN_MMD_PER_SIDE:
            return None, None
        return src, tgt

    def _compute_mmd_term(self, raw, embs, subj_ids, labels, mode):
        """Marginal or class-conditional MMD over the batch.

        marginal           : MMD(subject_0 embeddings, rest)            — existing.
        class_conditional  : mean over classes of MMD(subj_0|c, rest|c) — Phase 2,
                             aligns HIGH-HIGH and LOW-LOW across subjects so the
                             adversary cannot exploit class-correlated subject cues.
        Returns a scalar tensor, or None if no class/side has enough samples.
        """
        if mode == "class_conditional":
            # Split subjects into two balanced groups (parity of unique ids),
            # then align each engagement class across the two groups. Using
            # groups (not one-subject-vs-rest) keeps both sides populated even
            # after the per-class split.
            MIN_CC_PER_SIDE = 4
            uniq = subj_ids.unique()
            if len(uniq) < 2:
                return None
            groupA = uniq[0::2]
            in_a   = torch.isin(subj_ids, groupA)
            terms = []
            for c in labels.unique():
                cm  = labels == c
                src = embs[cm & in_a]
                tgt = embs[cm & ~in_a]
                if src.shape[0] >= MIN_CC_PER_SIDE and tgt.shape[0] >= MIN_CC_PER_SIDE:
                    terms.append(raw.mmd_loss(src, tgt))
            if not terms:
                return None
            return torch.stack(terms).mean()
        # marginal (default) — existing subject-vs-rest behaviour
        src, tgt = self._split_mmd_embs(embs, subj_ids)
        if src is None:
            return None
        return raw.mmd_loss(src, tgt)

    # ── GPU memory snapshot ──────────────────────────────────────────────────

    def _cuda_idx(self) -> int:
        return self.device.index if self.device.index is not None else 0

    def _cuda_active(self) -> bool:
        return torch.cuda.is_available() and self.device.type == "cuda"

    def _mem_gb(self) -> float:
        if not self._cuda_active():
            return 0.0
        return torch.cuda.memory_allocated(self._cuda_idx()) / 1e9

    def _peak_mem_gb(self) -> float:
        if not self._cuda_active():
            return 0.0
        return torch.cuda.max_memory_allocated(self._cuda_idx()) / 1e9

    # ── Train one epoch ──────────────────────────────────────────────────────

    def train_epoch(self, loader) -> dict:
        self.model.train()
        tracker      = MetricTracker()
        t0           = time.perf_counter()
        n_samples    = 0
        raw          = self._raw_model()
        # Reset peak memory tracker at epoch start (CUDA only)
        if self._cuda_active():
            torch.cuda.reset_peak_memory_stats(self._cuda_idx())

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch     = self._to_device(batch)
            n_samples += int(batch["label"].size(0))


            with autocast(self._amp_device, enabled=self.use_amp):
                out = self.model(
                    eeg_windows   = batch["eeg_windows"],
                    adj_matrices  = batch["adj_matrices"],
                    et_seq        = batch["et_seq"],
                    roi_vector    = batch["roi_vector"],
                    weighted_adjs = batch["weighted_adjs"],
                )

                src_emb, tgt_emb = self._split_mmd_embs(
                    out["eeg_emb"], batch["subject_id"]
                )

                losses = self.loss_fn(
                    model         = raw,
                    out           = out,
                    labels        = batch["label"],
                    roi_vector    = batch["roi_vector"],
                    weighted_adjs = batch["weighted_adjs"],
                    source_emb    = src_emb,
                    target_emb    = tgt_emb,
                )

                total_loss = losses["total"]

                # DANN adversarial loss (subject classifier with gradient reversal)
                if "dann_logits" in out and hasattr(self.loss_fn, 'lambda_dann'):
                    dann_loss = F.cross_entropy(out["dann_logits"], batch["subject_id"])
                    losses["dann"] = dann_loss.detach()
                    total_loss = total_loss + self.loss_fn.lambda_dann * dann_loss

                # MMD domain alignment (active in TRAINING — compute_loss keeps
                # mmd=0 on its training path, so the term is applied here).
                # mode="marginal": subject-vs-rest on eeg_emb.
                # mode="class_conditional": align HIGH-HIGH and LOW-LOW separately.
                if (hasattr(self.loss_fn, 'lambda_mmd')
                        and self.loss_fn.lambda_mmd > 0
                        and getattr(raw, 'mmd_loss', None) is not None
                        and "eeg_emb" in out):
                    mmd_term = self._compute_mmd_term(
                        raw, out["eeg_emb"], batch["subject_id"], batch["label"],
                        getattr(self.loss_fn, 'mmd_mode', 'marginal'),
                    )
                    if mmd_term is not None:
                        losses["mmd_train"] = mmd_term.detach()
                        total_loss = total_loss + self.loss_fn.lambda_mmd * mmd_term

                # Scale loss for gradient accumulation
                accum_loss = total_loss / self.grad_accum_steps

                # DataParallel gathers per-GPU scalars into a 1-D vector;
                # reduce to a scalar before backward so autograd is happy.
                if accum_loss.ndim > 0:
                    accum_loss = accum_loss.mean()

                # Shape / type guard
                assert torch.is_tensor(accum_loss), "accum_loss must be a Tensor"
                assert accum_loss.ndim == 0, \
                    f"accum_loss must be scalar, got shape {tuple(accum_loss.shape)}"

            if self.scaler:
                self.scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()

            # Step every grad_accum_steps micro-batches (or at epoch end)
            if (step + 1) % self.grad_accum_steps == 0:
                self._optimiser_step()

            # Metric tracking (no-grad, use un-scaled outputs)
            with torch.no_grad():
                probs = torch.softmax(out["logits"], dim=1)
                preds = probs.argmax(dim=1)
                tracker.update(
                    preds.cpu().numpy(),
                    probs.cpu().numpy(),
                    batch["label"].cpu().numpy(),
                    {k: v for k, v in losses.items()},
                )

        # Flush any leftover accumulated gradients
        if (len(loader)) % self.grad_accum_steps != 0:
            self._optimiser_step()

        # ReduceLROnPlateau is stepped in fit() after validation — not here.

        elapsed      = time.perf_counter() - t0
        throughput   = n_samples / max(elapsed, 1e-9)

        self._last_epoch_perf = {
            "duration_s"      : round(elapsed, 3),
            "throughput_sps"  : round(throughput, 1),
            "gpu_alloc_gb"    : round(self._mem_gb(),      3),
            "peak_mem_gb"     : round(self._peak_mem_gb(), 3),
        }

        return tracker.compute()

    def _optimiser_step(self) -> None:
        """Unscale → clip → step → zero_grad (AMP-aware)."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.optimizer.zero_grad()

    # ── Temperature calibration ──────────────────────────────────────────────

    def calibrate_temperature(self, val_loader) -> float:
        """
        Fit a scalar temperature T on val_loader by minimising NLL (Platt scaling).

        Small-sample guard: when val_loader has fewer than MIN_CAL_SAMPLES
        samples, skip LBFGS (it finds arbitrary T on <20 points) and return
        T=1.0.  Bounds are tightened to [T_MIN, T_MAX] to prevent the clamped-
        minimum (T=0.05) failure mode that causes ECE≈50% with tiny val sets.

        Returns the fitted temperature value.
        """
        MIN_CAL_SAMPLES = 20   # below this, calibration is statistically meaningless
        T_MIN           = 0.30  # prevents sharply overconfident predictions → ECE↑
        T_MAX           = 3.00  # prevents completely flat predictions

        self.model.eval()
        logits_buf: list = []
        labels_buf: list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                out = self.model(
                    eeg_windows   = batch["eeg_windows"],
                    adj_matrices  = batch["adj_matrices"],
                    et_seq        = batch["et_seq"],
                    roi_vector    = batch["roi_vector"],
                    weighted_adjs = batch["weighted_adjs"],
                )
                logits_buf.append(out["logits"])
                labels_buf.append(batch["label"])

        logits = torch.cat(logits_buf).detach()
        labels = torch.cat(labels_buf).detach()

        # Skip LBFGS when val set is too small — return uncalibrated (T=1.0)
        if logits.shape[0] < MIN_CAL_SAMPLES:
            self.temperature = 1.0
            return self.temperature

        T   = nn.Parameter(torch.ones(1, device=self.device))
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=500,
                                 tolerance_grad=1e-7, tolerance_change=1e-9)

        def _closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / T.clamp(min=T_MIN, max=T_MAX), labels)
            loss.backward()
            return loss

        opt.step(_closure)
        self.temperature = float(max(T_MIN, min(T_MAX, T.item())))
        return self.temperature

    # ── Evaluate one epoch ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """
        Evaluate on loader.  Logits are divided by self.temperature before
        softmax — call calibrate_temperature() first for calibrated probs.
        """
        self.model.eval()
        tracker = MetricTracker()

        for batch in loader:
            batch = self._to_device(batch)

            out = self.model(
                eeg_windows   = batch["eeg_windows"],
                adj_matrices  = batch["adj_matrices"],
                et_seq        = batch["et_seq"],
                roi_vector    = batch["roi_vector"],
                weighted_adjs = batch["weighted_adjs"],
            )

            probs = torch.softmax(out["logits"] / self.temperature, dim=1)
            preds = probs.argmax(dim=1)
            tracker.update(
                preds.cpu().numpy(),
                probs.cpu().numpy(),
                batch["label"].cpu().numpy(),
            )

        return tracker.compute()

    # ── Full training run ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        epochs      : int            = 150,
        verbose     : bool           = True,
        resume_path : Optional[Path] = None,
    ) -> list:
        """
        Train for up to *epochs* epochs with early stopping on val accuracy.

        Checkpoints are saved atomically to self.ckpt_path as full training
        state (model + optimizer + scheduler + scaler + counters).

        Parameters
        ----------
        resume_path : path to a checkpoint to resume from.  Supports both
                      full-state checkpoints (new format) and legacy
                      weights-only checkpoints.  When a full-state checkpoint
                      is loaded the training loop continues from the next epoch
                      with all states (LR, patience counters, etc.) restored.

        Returns
        -------
        history : list of dicts (one per epoch, train + val metrics)
        """
        best_val_bal = -1.0   # balanced_acc: 0.5 for any constant predictor
        no_improve   = 0
        start_epoch  = 1
        history      = []
        self.perf_log = []
        cumulative_s  = 0.0

        # ── Resume from checkpoint ───────────────────────────────────────────
        if resume_path is not None:
            rp = Path(resume_path)
            if rp.exists():
                resume_state = self._load_checkpoint(rp)
                if resume_state is not None:
                    start_epoch  = resume_state["epoch"] + 1
                    best_val_bal = resume_state["best_val_bal"]
                    no_improve   = resume_state["no_improve"]
                    if verbose:
                        print(f"[INFO] Resumed from epoch {resume_state['epoch']}",
                              flush=True)
                else:
                    if verbose:
                        print(f"[INFO] Loaded legacy checkpoint {rp.name} "
                              f"— starting from epoch 1", flush=True)
            else:
                if verbose:
                    print(f"[WARN] Resume checkpoint not found: {rp} "
                          f"— starting fresh", flush=True)

        # Log GPU config at fold start (CUDA only)
        if verbose and self._cuda_active():
            from utils.gpu import log_gpu_memory
            log_gpu_memory(self.device, prefix="[fold start]", reset_peak=True)

        for epoch in range(start_epoch, epochs + 1):
            # Warm auxiliary losses: ramp from 0→1 over the first 40% of
            # training.  Classification loss is always at full weight.  This
            # prevents auxiliary terms (contrastive, ROI, MMD) from dominating
            # early gradients when the model hasn't yet learned basic features.
            aux_ramp = min(1.0, (epoch - 1) / max(1, int(epochs * 0.40)))
            if hasattr(self.loss_fn, 'lambda_contrast'):
                self.loss_fn._aux_ramp = aux_ramp  # read in forward() below

            # DANN GRL alpha: standard schedule 2/(1+exp(-10p))-1, ramps 0→1
            p = (epoch - 1) / max(1, epochs - 1)
            dann_alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            _raw = self._raw_model()
            if hasattr(_raw, 'grl'):
                _raw.grl.set_alpha(float(dann_alpha))

            train_m = self.train_epoch(train_loader)
            val_m   = self.evaluate(val_loader)

            perf = self._last_epoch_perf.copy()
            perf["epoch"]        = epoch
            cumulative_s        += perf["duration_s"]
            perf["cumulative_s"] = round(cumulative_s, 2)
            self.perf_log.append(perf)

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_m.items()})
            row.update({f"val_{k}"  : v for k, v in val_m.items()})
            history.append(row)

            val_acc     = val_m.get("accuracy",     0.0)
            val_f1      = val_m.get("f1",           0.0)
            val_bal_acc = val_m.get("balanced_acc", 0.0)

            # CosineAnnealingWarmRestarts steps by epoch (not by metric).
            self.scheduler.step()

            if val_bal_acc > best_val_bal + 1e-4:
                best_val_bal = val_bal_acc
                no_improve   = 0
                self._save_checkpoint(epoch, best_val_bal, no_improve)
            else:
                no_improve += 1

            current_lr = self.optimizer.param_groups[0]["lr"]

            if verbose and epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"train_acc={train_m.get('accuracy', 0):.4f}  "
                    f"val_bal={val_bal_acc:.4f}  "
                    f"val_f1={val_f1:.4f}  "
                    f"lr={current_lr:.2e}  "
                    f"t={perf['duration_s']:.1f}s  "
                    f"thr={perf['throughput_sps']:.0f}sps  "
                    f"mem={perf['peak_mem_gb']:.2f}GB  "
                    f"patience={no_improve}/{self.patience}",
                    flush=True,
                )

            # Require at least 20 epochs before early stopping fires so the
            # model has time to move past random-init before balanced_acc is used
            # as a gate.
            if epoch >= 20 and no_improve >= self.patience:
                if verbose:
                    print(
                        f"  [EarlyStop] stopped_epoch={epoch}  "
                        f"best_val_bal={best_val_bal:.4f}  "
                        f"patience_used={no_improve}/{self.patience}",
                        flush=True,
                    )
                break

        # Restore best weights (into raw model under DP).
        # Handles both new full-state and legacy weights-only checkpoints.
        if self.ckpt_path.exists():
            try:
                ckpt = torch.load(
                    self.ckpt_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except Exception:
                ckpt = torch.load(self.ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self._raw_model().load_state_dict(ckpt["model_state_dict"])
            else:
                self._raw_model().load_state_dict(ckpt)

        self.best_val_acc = val_m.get("accuracy",     0.0)
        self.best_val_f1  = val_m.get("f1",           0.0)
        self.best_val_bal = best_val_bal
        return history
