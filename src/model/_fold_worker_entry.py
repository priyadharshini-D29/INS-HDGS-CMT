"""
Minimal fold-parallel worker entry point.

MUST NOT import torch (or any module that imports torch) at module level.
The spawned subprocess imports this module to find the worker function;
any top-level torch import would initialise the CUDA context with all
GPUs visible, defeating per-GPU pinning.

GPU pinning strategy
--------------------
The parent (_run_fold_parallel in losocv.py) sets CUDA_VISIBLE_DEVICES to
the single GPU index *before* calling Process.start() so the child inherits
the correct value at exec time.  This function also sets it as a fallback for
direct invocations and to be explicit about intent.

Exported name:
  run_gpu_worker(gpu_id, fold_cfgs_for_gpu, result_list, lock)
    -- one process per GPU; handles all folds assigned to that GPU.
"""


def run_gpu_worker(gpu_id, fold_cfgs_for_gpu, result_list, lock):
    """
    Entry point for one GPU worker process.

    CUDA_VISIBLE_DEVICES is set BEFORE importing torch so the CUDA context
    is initialised with exactly one visible device (cuda:0 = physical GPU
    gpu_id).  The parent already set it before spawning, so this is
    belt-and-suspenders for robustness.
    """
    import os, sys, signal
    from pathlib import Path

    # Workers must not die on Ctrl-C / SIGINT from the terminal — background
    # nohup jobs can still receive SIGINT via multiprocessing infrastructure.
    # Ignoring it here keeps folds alive through parent-side interrupts.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    _pkg_root = str(Path(__file__).resolve().parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

    import torch
    n = torch.cuda.device_count()
    print(f"[GPU-WORKER] gpu_id={gpu_id}  device_count={n}  "
          f"(physical GPU {gpu_id} mapped to cuda:0)", flush=True)

    from utils.gpu import run_fold_on_device

    results = []
    for cfg in fold_cfgs_for_gpu:
        try:
            r = run_fold_on_device(**cfg)
            if r is not None:
                results.append(r)
                if cfg.get("verbose"):
                    print(f"  [GPU{gpu_id}] Fold {cfg['fold_no']:02d} "
                          f"Acc={r.get('accuracy', 0):.4f}  "
                          f"F1={r.get('f1', 0):.4f}", flush=True)
        except BaseException as exc:
            import traceback
            print(f"  [ERROR] GPU{gpu_id} Fold {cfg['fold_no']}: {exc}")
            traceback.print_exc()

    with lock:
        result_list += results
