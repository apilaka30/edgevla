import time
import torch
from torchvision import transforms
import numpy as np

import os
from pathlib import Path

from PIL import Image
from prismatic.models import load, load_vla
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM

import os
from pathlib import Path

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def benchmark_model(model, image: Image, instruction: str, processor=None, n_trials=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    
    # Warm-up
    print(f"Warming up the GPU...")
    for _ in range(10):
        if processor is not None: # can eventually be only predict_action() once edgevla is trained
            inputs = processor(instruction, image).to(device, dtype=torch.float16)
            a = model.predict_action(**inputs, do_sample=False)
            print(f"OpenVLA action prediction output: {a}")
        else:
            _ = model.generate(image=image, prompt_text=instruction)

    print(f"Starting benchmarking with {n_trials} trials...")
    timings = np.zeros((n_trials, 1))
    for it in range(n_trials):
        if processor is not None: # can eventually be only predict_action() once edgevla is trained
            starter.record()
            inputs = processor(instruction, image).to(device, dtype=torch.float16)
            _ = model.predict_action(**inputs, do_sample=False)
            ender.record()
        else:
            starter.record()
            _ = model.generate(image=image, prompt_text=instruction)
            ender.record()
        torch.cuda.synchronize()
        timings[it] = starter.elapsed_time(ender)

    mean_inf = np.sum(timings) / n_trials
    std_inf = np.std(timings)
    return mean_inf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edgevla_path = Path("/home/apilaka/edgevla/checkpoints/vlm/llava-lvis-lrv")
    edgevla = load(edgevla_path, hf_token=os.environ["HF_TOKEN"]).half().to(device).eval()
    for param in edgevla.parameters():
        assert param.dtype == torch.float16, f"Loaded EdgeVLA parameters not in half precision: {param}"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    openvla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-spatial",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    ).to(device).eval()

    for param in openvla.parameters():
        assert param.dtype == torch.float16, f"Loaded OpenVLA parameter not in half precision: {param}"

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b-finetuned-libero-spatial", trust_remote_code=True)

    image = Image.open("/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/imgs/sample_img1.png").convert("RGB")
    

    print("Benchmarking OpenVLA...")
    openvla_time = benchmark_model(openvla, image=image, processor=processor, instruction="In: What action should the robot take to Open cabinet door?\nOut:")
    print(f"OpenVLA average inference time: {openvla_time:.2f} ms")

    print("Benchmarking EdgeVLA...")
    edgevla_time = benchmark_model(edgevla, image=image, instruction="In: What action should the robot take to Open cabinet door?\nOut:")
    print(f"EdgeVLA average inference time: {edgevla_time:.2f} ms")

    print("\nSpeedup:")
    print(f"EdgeVLA is {openvla_time / edgevla_time:.2f}x faster than OpenVLA")

if __name__ == "__main__":
    main()
