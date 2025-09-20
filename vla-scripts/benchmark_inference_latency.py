import torch
import numpy as np

import os
from pathlib import Path

from PIL import Image

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def benchmark_model(model, image: Image, instruction: str, processor=None, n_trials=50, unnorm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    unnorm_key: str = "libero_spatial_no_noops" if unnorm else None
    # Warm-up
    print(f"Warming up the GPU...")
    inputs = processor(instruction, image).to(device, dtype=torch.bfloat16)
    for _ in range(10):
        _ = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    print(f"Starting benchmarking with {n_trials} trials...")
    timings = np.zeros((n_trials, 1))
    for it in range(n_trials):
        starter.record()
        _ = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        ender.record()
        torch.cuda.synchronize()
        timings[it] = starter.elapsed_time(ender)

    mean_inf = np.sum(timings) / n_trials
    # std_inf = np.std(timings)
    return mean_inf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    openvla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b-finetuned-libero-spatial",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
        ).to(device).eval()

    for param in openvla.parameters():
        assert param.dtype == torch.bfloat16, f"Loaded OpenVLA parameter not in half precision: {param}"

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial", trust_remote_code=True)

    image = Image.open("/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/imgs/sample_img1.png").convert("RGB")
    
    config: OpenVLAConfig = AutoConfig.from_pretrained("/home/apilaka/vla-ft-lambda/runs/edgevla+libero_spatial_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug", trust_remote_code=True)
    config.text_config.vocab_size = 32064  # Set the text vocabulary size to match the tokenizer's vocab size.

    edgevla = AutoModelForVision2Seq.from_pretrained(
        "/home/apilaka/vla-ft-lambda/runs/edgevla+libero_spatial_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug",
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()

    print("Benchmarking OpenVLA...")
    openvla_time = benchmark_model(openvla, image=image, processor=processor, instruction="In: What action should the robot take to Open cabinet door?\nOut:")
    print(f"OpenVLA average inference time: {openvla_time:.2f} ms")

    print("Benchmarking EdgeVLA...")
    edgevla_time = benchmark_model(edgevla, image=image, processor=processor, instruction="In: What action should the robot take to Open cabinet door?\nOut:", unnorm=True)
    print(f"EdgeVLA average inference time: {edgevla_time:.2f} ms")

    print("\nSpeedup:")
    print(f"EdgeVLA is {openvla_time / edgevla_time:.2f}x faster than OpenVLA")

if __name__ == "__main__":
    main()
