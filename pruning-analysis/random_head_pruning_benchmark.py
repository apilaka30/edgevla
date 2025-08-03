#!/usr/bin/env python3
"""
random_head_pruning_benchmark.py

Measure inference speedup of TinyLLaMA as you randomly prune attention heads.
"""

"""
To run this script:

python random_head_pruning_benchmark.py \
  --model_name your-llama-tiny-id \
  --rlds_env dm_control:cartpole_swingup_with_images \
  --prompt "Describe the state." \
  --max_prune 0.4 --step 0.1 --num_runs 5

"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast

# Optional: for sampling an RLDS datapoint
try:
    import rlds
except ImportError:
    rlds = None


def random_head_prune(model: LlamaForCausalLM, prune_pct: float):
    """
    Randomly prune `prune_pct` fraction of all attention heads in the model.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    total_heads = n_layers * n_heads
    n_prune = int(total_heads * prune_pct)

    # build list of (layer_idx, head_idx)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    # randomly sample heads to prune
    to_prune = random.sample(all_heads, k=n_prune)

    # group by layer
    prune_map = {}
    for layer_idx, head_idx in to_prune:
        prune_map.setdefault(layer_idx, set()).add(head_idx)

    # apply
    for layer_idx, heads in prune_map.items():
        # HF's LlamaAttention has prune_heads()
        model.model.layers[layer_idx].self_attn.prune_heads(heads)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # replace with your TinyLLaMA repo
        help="HuggingFace model ID for TinyLLaMA",
    )
    p.add_argument(
        "--rlds_env",
        type=str,
        default=None,
        help="(Optional) RLDS env, e.g. 'dm_control:cartpole_swingup_with_images'",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Prompt to the LLM if not using RLDS",
    )
    p.add_argument(
        "--max_prune",
        type=float,
        default=0.4,
        help="Maximum fraction of heads to prune",
    )
    p.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Fractional step between pruning levels",
    )
    p.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of inference runs to average timing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name, token=Path(".hf_token"))
    print(f"Retreived TinyLLama tokenizer from {args.model_name}")

    # build a single input
    if args.rlds_env and rlds is not None:
        # sample one transition
        ds = rlds.load_dataset(args.rlds_env, splits=["train"], shuffle_buffer_size=1000)
        sample = next(iter(ds))
        # ▶ map sample to text somehow; here we stringify the state vector
        text = f"State: {sample['observation']['state']}"
    else:
        text = args.prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)

    prune_levels = np.arange(0.0, args.max_prune + 1e-8, args.step)
    latencies = []
    # print(f"Testing Pruning levels: {(list)prune_levels * 100:.1f}%")
    for pct in prune_levels:
        # print(f"Testing latency for {pct*100:>5.1f}% pruning...")
        # reload model for a clean slate
        model = (
            LlamaForCausalLM.from_pretrained(args.model_name, token=Path(".hf_token"))
            .to(device)
            .eval()
        )
        random_head_prune(model, prune_pct=pct)

        # warm‐up
        with torch.no_grad():
            for _ in range(2):
                _ = model(**inputs)

        # timed runs
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(args.num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(**inputs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        latencies.append(avg)
        print(f"Pruned {pct*100:>5.1f}% heads → {avg*1000:7.2f} ms")

    # compute speedup relative to 0% prune
    speedups = latencies[0] / np.array(latencies)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(prune_levels * 100, speedups, marker="o")
    plt.xlabel("% of Attention Heads Pruned")
    plt.ylabel("Inference Speedup (×)")
    plt.title("TinyLLaMA: Speedup vs. Random Head Pruning")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_vs_prune.png")
    # plt.show()


if __name__ == "__main__":
    main()


"""

    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    print(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    print(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )"""