import torch
import torch.nn as nn

from transformers import LlamaForCausalLM

def get_llm(model_name, device_map="auto"):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = LlamaForCausalLM.from_pretrained(model_name, 
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            device_map=device_map,
                                            do_sample=False,
                                            temperature=1.0,
                                            top_p=1.0)
    model.seqlen = 2048
    model.name = model_name
    model.generate()

    return model


