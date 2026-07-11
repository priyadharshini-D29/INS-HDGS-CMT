"""
================================================================
GPU-Optimized Inference Engine for Boosted Predictions
================================================================
Handles:
  - Mixed-precision inference (FP16 for speed)
  - Batch processing with dynamic sizes
  - Multi-GPU distributed inference
  - Caching and memory-efficient processing
  - Real-time performance monitoring
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Dict, List, Optional, Tuple
import numpy as np
import time
from pathlib import Path


class GPUOptimizedInference:
    """Efficient GPU inference with calibration and ensemble support."""

    def __init__(
        self,
        device: torch.device,
        use_mixed_precision: bool = True,
        batch_size: int = 32,
        enable_cudnn_benchmarking: bool = True,
    ):
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.batch_size = batch_size
        
        if enable_cudnn_benchmarking and device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, less deterministic
        
        self.perf_log = {
            "total_time_s": 0,
            "total_samples": 0,
            "peak_memory_gb": 0,
            "avg_throughput_sps": 0,
        }

    def _get_peak_memory(self) -> float:
        """Get peak GPU memory in GB."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1e9
        return 0.0

    def reset_memory_stats(self):
        """Reset GPU memory tracking."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def infer_batch(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        temperature: float = 1.0,
        return_probabilities: bool = True,
    ) -> Dict:
        """
        Efficient inference on a batch with mixed precision.
        
        Returns:
          {
            "logits": shape (batch_size, n_classes),
            "probabilities": shape (batch_size, n_classes) if return_probabilities,
            "predictions": shape (batch_size,),
            "confidence": shape (batch_size,),
          }
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        t_start = time.perf_counter()
        self.reset_memory_stats()
        
        with torch.no_grad():
            with autocast("cuda" if self.use_mixed_precision else "cpu", enabled=self.use_mixed_precision):
                output = model(
                    batch["eeg_windows"],
                    batch["adj_matrices"],
                    batch["et_seq"],
                    batch["roi_vector"],
                    batch.get("weighted_adjs", batch["adj_matrices"]),
                )
                
                # Extract logits
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("cls_logits", output.get("out", None)))
                    if logits is None:
                        logits = output
                else:
                    logits = output
        
        # Apply temperature scaling
        logits_scaled = logits / temperature
        
        # Compute probabilities
        if return_probabilities:
            probs = F.softmax(logits_scaled, dim=1)
        else:
            probs = None
        
        # Get predictions and confidence
        predictions = torch.argmax(logits_scaled, dim=1)
        confidence, _ = torch.max(F.softmax(logits_scaled, dim=1), dim=1)
        
        t_elapsed = time.perf_counter() - t_start
        batch_size = batch["eeg_windows"].size(0)
        
        # Update performance log
        self.perf_log["total_time_s"] += t_elapsed
        self.perf_log["total_samples"] += batch_size
        self.perf_log["peak_memory_gb"] = max(
            self.perf_log["peak_memory_gb"],
            self._get_peak_memory()
        )
        self.perf_log["avg_throughput_sps"] = self.perf_log["total_samples"] / self.perf_log["total_time_s"]
        
        result = {
            "logits": logits_scaled.cpu().numpy(),
            "predictions": predictions.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "inference_time_ms": t_elapsed * 1000,
        }
        
        if probs is not None:
            result["probabilities"] = probs.cpu().numpy()
        
        return result

    def infer_dataset(
        self,
        model: nn.Module,
        dataloader,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Efficient inference on full dataset.
        
        Returns:
          {
            "all_logits": shape (n_samples, n_classes),
            "all_predictions": shape (n_samples,),
            "all_confidence": shape (n_samples,),
            "all_labels": shape (n_samples,) [if available in batch],
            "performance": dict with timing/memory stats,
          }
        """
        model.eval()
        
        all_logits = []
        all_preds = []
        all_confs = []
        all_labels = []
        
        t_start = time.perf_counter()
        
        for batch_idx, batch in enumerate(dataloader):
            result = self.infer_batch(model, batch, temperature=temperature)
            
            all_logits.append(result["logits"])
            all_preds.append(result["predictions"])
            all_confs.append(result["confidence"])
            
            # Collect labels if available
            if "label" in batch:
                labels = batch["label"].cpu().numpy() if isinstance(batch["label"], torch.Tensor) else batch["label"]
                all_labels.append(labels)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} — "
                      f"Inference: {result['inference_time_ms']:.1f} ms")
        
        t_total = time.perf_counter() - t_start
        
        # Concatenate results
        result = {
            "all_logits": np.concatenate(all_logits),
            "all_predictions": np.concatenate(all_preds),
            "all_confidence": np.concatenate(all_confs),
            "performance": {
                "total_time_s": t_total,
                "n_samples": self.perf_log["total_samples"],
                "throughput_sps": self.perf_log["total_samples"] / t_total,
                "peak_memory_gb": self.perf_log["peak_memory_gb"],
                "avg_batch_time_ms": (t_total / len(dataloader)) * 1000,
            }
        }
        
        if all_labels:
            result["all_labels"] = np.concatenate(all_labels)
        
        return result


class CalibratedModelWrapper(nn.Module):
    """Wraps a model with temperature scaling."""

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, eeg_windows, adj_matrices, et_seq, roi_vector, weighted_adjs):
        output = self.model(eeg_windows, adj_matrices, et_seq, roi_vector, weighted_adjs)
        
        if isinstance(output, dict):
            logits = output.get("logits", output.get("cls_logits", output.get("out")))
            if logits is not None:
                output["logits"] = logits / self.temperature
        else:
            output = output / self.temperature
        
        return output

    def set_temperature(self, temp: float):
        self.temperature.copy_(torch.tensor(temp, device=self.temperature.device))


class MultiGPUEnsembleInference:
    """Distributes ensemble inference across multiple GPUs."""

    def __init__(self, devices: List[torch.device]):
        self.devices = devices
        self.model_cache = {}

    def load_model_to_gpu(self, model: nn.Module, gpu_id: int) -> nn.Module:
        """Load model to specific GPU."""
        device = self.devices[gpu_id]
        return model.to(device)

    def infer_on_gpu_pool(
        self,
        model_list: List[nn.Module],
        dataloader,
        temperatures: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Distribute ensemble models across GPUs and aggregate predictions.
        
        Returns:
          (all_logits, all_confidences, all_uncertainties)
        """
        if temperatures is None:
            temperatures = [1.0] * len(model_list)
        
        # Distribute models to GPUs
        models_on_gpus = []
        for idx, model in enumerate(model_list):
            gpu_id = idx % len(self.devices)
            model_gpu = self.load_model_to_gpu(model, gpu_id)
            models_on_gpus.append((model_gpu, self.devices[gpu_id], temperatures[idx]))
        
        all_logits_list = []
        all_uncertainties_list = []
        
        for batch in dataloader:
            batch_logits = []
            
            for model, device, temp in models_on_gpus:
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.no_grad():
                    with autocast("cuda"):
                        output = model(
                            batch_device["eeg_windows"],
                            batch_device["adj_matrices"],
                            batch_device["et_seq"],
                            batch_device["roi_vector"],
                            batch_device.get("weighted_adjs", batch_device["adj_matrices"]),
                        )
                        
                        logits = output.get("logits", output.get("cls_logits", output)) if isinstance(output, dict) else output
                        logits_scaled = logits / temp
                        batch_logits.append(logits_scaled.cpu())
            
            # Ensemble average
            ensemble_logits = torch.stack(batch_logits).mean(dim=0)
            all_logits_list.append(ensemble_logits)
            
            # Uncertainty: variance across ensemble
            logits_var = torch.stack(batch_logits).var(dim=0).mean(dim=1)
            all_uncertainties_list.append(logits_var)
        
        all_logits = np.concatenate([l.numpy() for l in all_logits_list])
        all_uncertainties = np.concatenate([u.numpy() for u in all_uncertainties_list])
        all_confidence = np.max(F.softmax(torch.from_numpy(all_logits), dim=1).numpy(), axis=1)
        
        return all_logits, all_confidence, all_uncertainties


def benchmark_inference(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_runs: int = 3,
) -> Dict:
    """Benchmark inference performance."""
    model.eval()
    
    times = []
    
    for run in range(n_runs):
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t_start = time.perf_counter()
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                _ = model(batch["eeg_windows"], batch["adj_matrices"], batch["et_seq"], 
                         batch["roi_vector"], batch.get("weighted_adjs", batch["adj_matrices"]))
        
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t_elapsed = time.perf_counter() - t_start
        times.append(t_elapsed)
    
    return {
        "mean_time_s": np.mean(times),
        "std_time_s": np.std(times),
        "min_time_s": np.min(times),
        "max_time_s": np.max(times),
        "throughput_sps": len(dataloader.dataset) / np.mean(times),
    }
