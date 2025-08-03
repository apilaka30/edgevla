import torch
import accelerate
from PIL import Image
import numpy as np

@torch.no_grad()
def test_latency(model, generation, image: Image, prompt: str):

    if (generation) :
        # setting for token generation
        generation_length = 128
        prompt_length = 64
        batch_size = 64
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        model.config.use_cache = True
        model.generation_config.use_cache = True
        iteration = 10

        timings = np.zeros((iteration, 1))

        # dummy inference
        for _ in range(10):
            _ = model.generate(image=image, prompt_text=prompt)

        # latency for 10 iterations
        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in range(iteration):
            starter.record()
            model.generate(image=image, prompt_text=prompt, min_new_tokens=generation_length, max_new_tokens=generation_length)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
            print(f"Latency for iteration {i+1}: {timings[i][0]} ms")

    else :
        # setting for prompt processing
        batch_size = 1
        model.config.use_cache = False
        model.generation_config.use_cache = False
        iteration = 50

        # make dummy input for module.weight shape
        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(model.device).contiguous()
        
        # dummy inference
        model(random_input)

        # latency for 50 iterations
        starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        for i in range(iteration):
            model(random_input)
        ender.record()
        torch.cuda.synchronize()

    mean_latency = np.sum(timings)/iteration

    return mean_latency