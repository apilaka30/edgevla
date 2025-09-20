#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile an EdgeVLA / OpenVLA-style model:
  1) Count parameters (total / trainable)
  2) Compute parameter magnitude statistics & histograms
  3) Compute checkpoint size on disk (GiB)

Usage examples:
  python edgevla_profile.py \
      --checkpoint /path/to/edgevla-checkpoint \
      --dtype bfloat16 \
      --attn_impl flash_attention_2 \
      --set vocab_size=32064 \
      --set vision_config.image_size=256

  python edgevla_profile.py --checkpoint /path/to/hf_hub_dir
"""

import os
import re
import json
import math
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np

# --- Register OpenVLA HF classes (adjust imports to your local package layout) ---
from transformers import (
    AutoConfig, AutoImageProcessor, AutoProcessor,
    AutoModelForVision2Seq,
)

# If your project exposes these under prismatic.extern.hf.* (OpenVLA):
try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
except Exception as e:
    # If not available, model loading may still work if the checkpoint already includes `auto_map` / trust_remote_code
    print(f"[warn] Could not register OpenVLA classes ({e}). "
          "If your checkpoint has trust_remote_code + auto_map, loading can still succeed.")


# -------------------------
# Utilities
# -------------------------

def parse_overrides(kvs: List[str]) -> Dict[str, Any]:
    """
    Parse --set key=value pairs, with dotted keys for nested fields.
    Values are parsed as int, float, bool, or left as string.
    """
    def parse_value(v: str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v

    out: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        out[k.strip()] = parse_value(v.strip())
    return out


def setattr_deep(obj, dotted_key: str, value):
    """
    Set a (possibly nested) attribute on a HF config object via dotted path.
    Example: setattr_deep(config, "vision_config.image_size", 256)
    """
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        # HF configs sometimes store nested as attributes or dicts
        if hasattr(cur, p):
            cur = getattr(cur, p)
        elif isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise AttributeError(f"Path segment '{p}' not found in config for key '{dotted_key}'")
    last = parts[-1]
    if hasattr(cur, last):
        setattr(cur, last, value)
    elif isinstance(cur, dict):
        cur[last] = value
    else:
        raise AttributeError(f"Leaf '{last}' not found in config for key '{dotted_key}'")


def dtype_from_str(s: str):
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def human_gib(nbytes: int) -> float:
    return nbytes / (1024**3)


def total_checkpoint_size_gib(checkpoint_path: str) -> Tuple[float, Dict[str, float]]:
    """
    Sum common weight file formats under a checkpoint directory.
    Includes *.safetensors, *.bin, *.pt, *.pth, *.msgpack (GGUF not included by default).
    """
    exts = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.msgpack"]
    sizes: Dict[str, float] = {}
    total = 0
    for ext in exts:
        matched = glob.glob(os.path.join(checkpoint_path, "**", ext), recursive=True)
        bytes_sum = sum(os.path.getsize(f) for f in matched if os.path.isfile(f))
        sizes[ext] = human_gib(bytes_sum)
        total += bytes_sum
    return human_gib(total), sizes


# -------------------------
# Magnitude Profiling
# -------------------------

def online_log_histogram(
    tensor_iter,
    log10_min: float = -12.0,
    log10_max: float = +1.0,
    num_bins: int = 60,
):
    """
    Accumulate a histogram over |param| using logarithmic bins for |w|.
    Returns:
        dict with:
          - counts (num_bins)
          - edges (num_bins+1, log10 space)
          - zero_count
          - tiny_count (|w| < 10^log10_min but > 0)
          - huge_count (|w| > 10^log10_max)
          - total_elems
          - order_of_mag_counts (dict: exponent -> count)
          - percentiles (abs)
          - summary stats (l1_mean, l2_rms, max_abs)
    """
    # Prepare log10 bin edges
    edges = np.linspace(log10_min, log10_max, num_bins + 1, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)

    zero_count = 0
    tiny_count = 0
    huge_count = 0
    total_elems = 0

    # For additional stats
    abs_samples_for_pct = []
    # For memory safety, we'll reservoir-sample a subset for percentiles
    sample_cap = 5_000_000

    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0

    order_mag_counts: Dict[int, int] = {}

    rng = np.random.default_rng(0xC0FFEE)

    for t in tensor_iter:
        if t.numel() == 0:
            continue
        a = t.detach().float().abs().view(-1).cpu()

        total_elems += a.numel()
        zero_mask = (a == 0)
        zero_count += int(zero_mask.sum().item())
        nz = a[~zero_mask]

        if nz.numel() > 0:
            # Stats
            sum_abs += float(nz.sum().item())
            sum_sq += float((nz * nz).sum().item())
            max_abs = max(max_abs, float(nz.max().item()))

            # Order-of-magnitude buckets: floor(log10(|w|))
            with torch.no_grad():
                log10_vals = torch.log10(nz)
            exponents = torch.floor(log10_vals).to(torch.int64).cpu().numpy()
            for e in np.unique(exponents):
                order_mag_counts[int(e)] = order_mag_counts.get(int(e), 0) + int((exponents == e).sum())

            # Histogram accumulation on log10(|w|)
            log10_vals_np = log10_vals.cpu().numpy()
            # tiny / huge tracking
            tiny_count += int((log10_vals_np < log10_min).sum())
            huge_count += int((log10_vals_np > log10_max).sum())

            # Clip for binning into [log10_min, log10_max]
            clipped = np.clip(log10_vals_np, log10_min, log10_max)
            hist, _ = np.histogram(clipped, bins=edges)
            counts += hist

            # Reservoir-sample for percentiles
            take = nz.cpu().numpy()
            if len(abs_samples_for_pct) < sample_cap:
                need = sample_cap - len(abs_samples_for_pct)
                if take.size <= need:
                    abs_samples_for_pct.extend(take.tolist())
                else:
                    idx = rng.choice(take.size, size=need, replace=False)
                    abs_samples_for_pct.extend(take[idx].tolist())
            else:
                # Replace randomly with low probability (skip to keep things simple & fast)
                pass

    l1_mean = sum_abs / max(1, (total_elems - zero_count))
    l2_rms = math.sqrt(sum_sq / max(1, (total_elems - zero_count)))

    percentiles = {}
    if abs_samples_for_pct:
        s = np.array(abs_samples_for_pct, dtype=np.float64)
        for p in (0.0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100.0):
            percentiles[str(p)] = float(np.percentile(s, p))

    return {
        "counts": counts.tolist(),
        "edges_log10": edges.tolist(),
        "zero_count": int(zero_count),
        "tiny_count": int(tiny_count),
        "huge_count": int(huge_count),
        "total_elems": int(total_elems),
        "order_of_mag_counts": {str(k): int(v) for k, v in sorted(order_mag_counts.items())},
        "summary": {
            "nonzero_l1_mean": float(l1_mean),
            "nonzero_l2_rms": float(l2_rms),
            "max_abs": float(max_abs),
            "zero_frac": float(zero_count / max(1, total_elems)),
        },
        "percentiles_abs": percentiles,
    }


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="EdgeVLA / OpenVLA Profiler")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to HF checkpoint dir (or HF cache dir).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to instantiate on (profiling runs on CPU).")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"])
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        help="Passed to from_pretrained(attn_implementation=...).")
    parser.add_argument("--low_cpu_mem_usage", action="store_true",
                        help="Pass low_cpu_mem_usage=True to HF loader.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True (usually needed for OpenVLA).")
    parser.add_argument("--set", dest="overrides", action="append",
                        help="Config override(s) like key=value (supports dotted keys). Use multiple --set flags.")

    # Histogram controls
    parser.add_argument("--log10_min", type=float, default=-12.0,
                        help="Lower edge (log10) for magnitude histogram bins (default: -12).")
    parser.add_argument("--log10_max", type=float, default=+1.0,
                        help="Upper edge (log10) for magnitude histogram bins (default: +1).")
    parser.add_argument("--num_bins", type=int, default=60,
                        help="Number of log-magnitude bins (default: 60).")
    parser.add_argument("--json_out", type=str, default="edgevla_profile.json",
                        help="Path to write JSON report.")
    args = parser.parse_args()

    ckpt = args.checkpoint
    dtype = dtype_from_str(args.dtype)

    # --- Load & modify config BEFORE weights ---
    print(f"[info] Loading config from: {ckpt}")
    config = AutoConfig.from_pretrained(
        ckpt,
        trust_remote_code=args.trust_remote_code or True,  # EdgeVLA/OpenVLA typically needs this
    )

    overrides = parse_overrides(args.overrides)
    for k, v in overrides.items():
        setattr_deep(config, k, v)
        print(f"[info] Set config.{k} = {v}")

    # --- Instantiate model with modified config ---
    print(f"[info] Instantiating model with dtype={dtype}, device={args.device} ...")
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt,
        config=config,                      # <- use modified config
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code or True,
        # You can also pass load_in_8bit/4bit=False here if desired
    )

    if args.device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")

    # --- Parameter counts ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Histogram / stats over magnitudes (always do on CPU to avoid VRAM) ---
    model_cpu = model.to("cpu")
    with torch.no_grad():
        mag_report = online_log_histogram(
            (p for p in model_cpu.parameters()),
            log10_min=args.log10_min,
            log10_max=args.log10_max,
            num_bins=args.num_bins,
        )

    # --- Checkpoint size (GiB) ---
    total_gib, by_ext = total_checkpoint_size_gib(ckpt)

    # --- Compose report ---
    report = {
        "checkpoint_path": os.path.abspath(ckpt),
        "dtype_loaded": str(dtype),
        "attn_implementation": args.attn_impl,
        "param_counts": {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "total_params_millions": round(total_params / 1e6, 3),
            "trainable_params_millions": round(trainable_params / 1e6, 3),
        },
        "magnitude_report": mag_report,
        "checkpoint_size_gib": round(total_gib, 4),
        "checkpoint_size_breakdown_gib": {k: round(v, 4) for k, v in by_ext.items()},
        "config_snippet": config.to_dict() if hasattr(config, "to_dict") else None,
    }

    # Print concise summary
    print("\n=== EdgeVLA / OpenVLA Profile Summary ===")
    print(f"Checkpoint:   {report['checkpoint_path']}")
    print(f"Params:       {report['param_counts']['total_params_millions']} M "
          f"(trainable {report['param_counts']['trainable_params_millions']} M)")
    print(f"Zeros:        {mag_report['summary']['zero_frac']*100:.4f}% of parameters")
    print(f"Abs max:      {mag_report['summary']['max_abs']:.6g}")
    print(f"Abs L1 mean:  {mag_report['summary']['nonzero_l1_mean']:.6g}")
    print(f"Abs L2 RMS:   {mag_report['summary']['nonzero_l2_rms']:.6g}")
    pcts = mag_report["percentiles_abs"]
    if pcts:
        print(f"Abs percentiles: p50={pcts.get('50', float('nan')):.3e}, "
              f"p90={pcts.get('90', float('nan')):.3e}, "
              f"p99={pcts.get('99', float('nan')):.3e}")
    print(f"Checkpoint size: {report['checkpoint_size_gib']:.4f} GiB "
          f"(by ext: {report['checkpoint_size_breakdown_gib']})")

    # Write full JSON
    out_path = Path(args.json_out)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[info] Wrote detailed JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
