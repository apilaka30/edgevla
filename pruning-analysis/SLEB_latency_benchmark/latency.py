import torch
# import fire
import numpy as np
import os
from pathlib import Path
from prismatic.models import load
import copy

from block_remove import block_remove
import model_utils
import latency_utils
import matplotlib.pyplot as plt
from PIL import Image

def run(
    model: any = None,
    removal_ratio: float = 0.2,
    generation: bool = True,
    result_folder: str ='sleb_results',
    result_file: str = 'latency.txt',
    num_trials: int = 10,
    dense_latency: float = None,
    sample_img: Image = None,
    sample_prompt: str = "What action should the robot take to pick up the object?",
) -> tuple:
    gpu_num = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    num_of_blocks = model.llm_backbone.llm.config.num_hidden_layers
    num_removal = int(np.ceil(num_of_blocks * removal_ratio))
    removal_list = [i+1 for i in range(num_removal)]
    
    print("==================================================")
    print("Experiment Environment")
    print(f"Current GPU: {gpu_name}")
    print(f"# GPU: {str(gpu_num)}")
    print(f"Model Name: EdgeVLA")
    print(f"Infernce type : {'Token Generation' if generation else 'Prompt Processing'}")
    print("==================================================")

    # # latency for dense model
    # dense_trials = np.empty(num_trials)
    # for i in range(num_trials):
    #     dense_trials[i] = latency_utils.test_latency(model, generation)
    # dense_latency = np.mean(dense_trials)
    # print(f"Dense Latency (avg over {num_trials} trials): {dense_latency:.2f}ms")
    dense_llm = copy.deepcopy(model.llm_backbone.llm)
    # latency for sleb model
    if dense_latency is not None:
        model.llm_backbone.llm = block_remove(model.llm_backbone.llm, removal_list)

    
    sleb_latency = latency_utils.test_latency(model, generation, sample_img, sample_prompt)
    print(f"SLEB {removal_ratio} Latency (avg over {num_trials} trials): {sleb_latency:.2f}ms")

    # is the first 0.0% pruning run
    if dense_latency is None:
        dense_latency = sleb_latency

    speedup = dense_latency / sleb_latency
    print(f"Speedup: x{speedup:.2f}")
    print("==================================================")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)

    # save log
    with open(result_path, "a") as file: 
        file.write(f"Current GPU: {gpu_name}")
        file.write(", ")
        file.write(f"# GPU: {str(gpu_num)}")
        file.write(", ")
        file.write(f"Model Name: EdgeVLA")
        file.write(", ")
        file.write(f"Infernce type : {'Token Generation' if generation else 'Prompt Processing'}")
        file.write(", ")
        file.write(f"Dense Latency: {dense_latency:.2f}ms")
        file.write(", ")
        file.write(f"SLEB {removal_ratio} Latency: {sleb_latency:.2f}ms")
        file.write(", ")
        file.write(f"Speedup: x{speedup:.2f}")
        file.write("\n")

    model.llm_backbone.llm = dense_llm #return LLM to original state
    return (dense_latency, sleb_latency, speedup)

if __name__ == '__main__':
#     # fire.Fire(run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_path = Path("/home/apilaka/edgevla/checkpoints/vlm/llava-lvis-lrv")
    vlm = load(vlm_path, hf_token=os.environ["HF_TOKEN"], load_for_training=False).to(device).eval()


    img = Image.open("/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/imgs/sample_img1.png").convert("RGB")
    instruction = "What should the robot do to Open cabinet door?"

    dlat = None

    step = 0.05
    max_ratio = 0.3
    # Run the experiment for different removal ratios
    # from 0 to max_ratio with the specified step size
    prune_ratios = np.arange(0, max_ratio + step, step)

    # Initialize arrays to store results
    dense_latencies = np.zeros_like(prune_ratios)
    sleb_latencies = np.zeros_like(prune_ratios)
    speedups = np.zeros_like(prune_ratios)
    
    for i, ratio in enumerate(prune_ratios):
        (dense_latency, sleb_latency, speedup) = run(
            model=vlm,
            removal_ratio=ratio,
            generation=True,
            result_folder='/home/apilaka/edgevla/openvla/pruning-analysis/sleb_results_V100_vlm',
            result_file='vlm_latency.txt',
            num_trials=10,
            dense_latency=dlat,
            sample_img=img,
            sample_prompt=instruction,
        )
        dense_latencies[i] = dense_latency
        sleb_latencies[i] = sleb_latency
        speedups[i] = speedup

        dlat = dense_latency if dlat is None else dlat


    plt.figure(figsize=(6,4))
    plt.plot(prune_ratios, speedups, marker="o")
    plt.xlabel("% Blocks Pruned")
    plt.ylabel("Avg Inference Speedup (X)")
    plt.title("EdgeVLA SLEB Depth-Pruning Ratio vs Inference Speedup")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/apilaka/edgevla/openvla/pruning-analysis/sleb_results_V100_vlm/sleb_block_prune_speedup.png")


    plt.figure(figsize=(6,4))
    plt.plot(prune_ratios, sleb_latencies, marker="o")
    plt.xlabel("% Blocks Pruned")
    plt.ylabel("Avg Inference Latency (ms)")
    plt.title("EdgeVLA SLEB Depth-Pruning Ratio vs. Generation Latency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/apilaka/edgevla/openvla/pruning-analysis/sleb_results_V100_vlm/sleb_block_prune_latencies.png")