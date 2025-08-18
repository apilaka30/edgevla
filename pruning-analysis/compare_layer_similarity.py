# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
# import numpy as np
# import argparse
# import os
# from pathlib import Path

# from PIL import Image
# from prismatic.models import load

# from torch.nn.utils.rnn import pad_sequence
# import random
# from CKA import linear_CKA

# random.seed(42)  # For reproducibility

# @torch.no_grad()
# def get_llm_block_outputs(model, pixel_values, input_ids, attention_mask):
#     outputs = []

#     # Single forward pass through the full VLM/VLA model
#     autocast_dtype = model.llm_backbone.half_precision_dtype
#     model_output = None
#     with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
#         model_output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
#     # Extract hidden states from all transformer blocks (shape: (num_layers, batch_size, seq_len, hidden_dim))
#     outputs = model_output.hidden_states

#     return outputs


# def compute_cosine_matrix(outputs):
#     n = len(outputs)
#     # print(f"Hidden state dimension tensors: {outputs[0].shape}")
#     sim_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             token_idx = random.randint(0, outputs[i].shape[1]-1)  # Randomly select a token index
            
            
#             v1 = outputs[i][0][token_idx].cpu() # extract hidden state for the selected token (token_idx) at the layer i (specify batch idx 0)
#             v2 = outputs[j][0][token_idx].cpu() # extract hidden state for the selected token (token_idx) at the layer j (specify batch idx 0)

#             # calculate cosine similarity
#             sim_matrix[i, j] = F.cosine_similarity(v1, v2, dim=0).item()
#     return sim_matrix


# def plot_upper_triangle_similarity_matrix(
#     sim_matrix,
#     title,
#     save_path,
#     cmap="Greens",
#     vmin=0.0,
#     vcenter=0.5,
#     vmax=1.0,
#     lower_color="lightgray",
#     grid_color="white",
#     grid_linewidth=0.25,
#     show_values=False
# ):
#     """
#     Plot only the upper triangle of a similarity matrix using the PiYG colormap,
#     and fill the lower triangle with a gray background like in the SLEB figure.
#     """
#     n = sim_matrix.shape[0]

#     # Mask lower triangle (i > j)
#     upper_mask = np.tril(np.ones_like(sim_matrix, dtype=bool), k=-1)
#     upper_data = np.ma.masked_array(sim_matrix, mask=upper_mask)

#     # Create gray background mask for lower triangle
#     background = np.full_like(sim_matrix, fill_value=np.nan, dtype=np.float32)
#     background[upper_mask] = 0.5  # middle value (centered gray in grayscale map)
#     background = np.ma.masked_invalid(background)

#     # Plot setup
#     plt.figure(figsize=(8, 7))

#     # Gray lower triangle
#     plt.imshow(
#         background,
#         cmap=plt.cm.Greys,
#         vmin=0,
#         vmax=1,
#         interpolation="nearest"
#     )

#     # Colored upper triangle with PiYG diverging colormap
#     norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
#     im = plt.imshow(
#         upper_data,
#         cmap=cmap,
#         # norm=norm,
#         interpolation="nearest"
#     )

#     plt.colorbar(im, label="Cosine similarity")
#     plt.title(title)
#     plt.xlabel("Layer j Index")
#     plt.ylabel("Layer i Index")

#     # Grid
#     plt.gca().set_xticks(np.arange(-0.5, n, 1), minor=True)
#     plt.gca().set_yticks(np.arange(-0.5, n, 1), minor=True)
#     plt.grid(True, which="minor", color=grid_color, linewidth=grid_linewidth)
#     plt.tick_params(axis='both', which='major', labelsize=8)

#     # Labels and layout
#     plt.title(title)
#     plt.xlabel("Layer j Index")
#     plt.ylabel("Layer i Index")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


# def build_prompt_and_inputs(instruction, image_path, base_tokenizer, image_transform, prompt_builder_fn):
#     """Mimics RLDSBatchTransform for a single image + instruction pair."""

#     # Language prompt builder
#     prompt_builder = prompt_builder_fn("openvla")
#     lang = instruction.lower()
#     prompt_builder.add_turn("human", f"What action should the robot take to {lang}?")

#     prompt = prompt_builder.get_prompt()
#     input_ids = base_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids

#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=base_tokenizer.pad_token_id)[:, : base_tokenizer.model_max_length]

#     attention_mask = input_ids.ne(base_tokenizer.pad_token_id)

#     # Image processing
#     img = Image.open(image_path).convert("RGB")
#     pixel_values = image_transform(img)
#     pixel_values['dino'] = pixel_values['dino'].unsqueeze(0)  # add batch dimension
#     pixel_values['siglip'] = pixel_values['siglip'].unsqueeze(0)  # add batch dimension
#     # print("Input IDs shape:", input_ids.shape)
#     # print(pixel_values['dino'].shape, pixel_values['siglip'].shape)

#     return dict(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         pixel_values=pixel_values
#     )


# def main(args):
#     # Get device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     vlm_path = Path("/home/apilaka/edgevla/checkpoints/vlm/llava-lvis-lrv")

#     # Load model and tokenizer
#     vlm = load(vlm_path, hf_token=os.environ["HF_TOKEN"], load_for_training=False).to(device).eval()
#     tokenizer = vlm.llm_backbone.get_tokenizer()
#     image_transform = vlm.vision_backbone.get_image_transform()
#     # model_post = load_vla(vla_path, hf_token=os.environ["HF_TOKEN"], load_for_training=False).eval()

#     agg_type = 'avg'  # ('min' or 'avg')
#     # If agg_type is 'min', we compute the minimum similarity across samples (more strict)
#     agg_sim_matrix = np.full((23, 23), fill_value=1.0) if agg_type == 'min' else np.zeros((23, 23))
#     num_samples = 350
#     with open("/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/sample_instructions.txt", "r") as f:
#         for i, line in enumerate(f, start=1):
#             if i > num_samples:
#                 break
#             text_prompt = line.strip()
#             image_path = f"/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/imgs/sample_img{i}.png"
#             # Build inputs
#             inputs = build_prompt_and_inputs(
#                 instruction=text_prompt,
#                 image_path=image_path,
#                 base_tokenizer=tokenizer,
#                 image_transform=image_transform,
#                 prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn
#             )
#             # Move inputs to device
#             for k in inputs:
#                 if k == "pixel_values":
#                     inputs[k] = {key: v.to(device) for key, v in inputs[k].items()}
#                 else:
#                     inputs[k] = inputs[k].to(device)
            

#             print(f"Computing similarity...[{i}/{num_samples}]")
#             out_pre = get_llm_block_outputs(vlm, **inputs)
#             sim_pre = compute_cosine_matrix(out_pre)

#             # Aggregate similarity matrices
#             agg_sim_matrix = np.minimum(agg_sim_matrix, sim_pre) if agg_type == 'min' else agg_sim_matrix + sim_pre*(1/num_samples)
    
#     # Save and plot results
#     np.save(os.path.join(args.output_dir, "cosine_pre.npy"), agg_sim_matrix)
#     plot_upper_triangle_similarity_matrix(agg_sim_matrix, "Pre-Finetuning Instruction Grounding Cosine Similarity", os.path.join(args.output_dir, "cosine_pre.png"))



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_path", type=str, default="/home/apilaka/edgevla/openvla/pruning-analysis/robotics_sample/sample_image.png",
#                         help="Path to input image")
#     parser.add_argument("--text_prompt", type=str, default="What",
#                         help="Language instruction to build the prompt")
#     parser.add_argument("--output_dir", type=str, default="./cosine_similarity_results_robotics")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     main(args)

#!/usr/bin/env python
"""
Cosine‐similarity redundancy analysis for Prismatic VLM/VLA.

* `--mode prefix`  : original “instruction grounding” analysis
* `--mode action`  : 7-DoF autoregressive action-token analysis
"""

import argparse, os, random
from pathlib import Path
from PIL import Image

import torch, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from prismatic.models import load, load_vla
from torch.nn.utils.rnn import pad_sequence

random.seed(42)
torch.manual_seed(42)


# --------------------------------------------------------------------------- #
#  Cosine-similarity helpers
# --------------------------------------------------------------------------- #
def cosine_matrix_single_token(hidden_states, token_idx=-1):
    """
    hidden_states : tuple[( B (batch_size), S (seq_length), D (hidden_dim_size) )]  length = L+1 (includes embedding)
    token_idx     : int  (index into sequence dim S)
    returns       : (L, L) numpy  cosine-sim matrix between all block outputs
    """
    layers = hidden_states[1:]                   # drop embedding layer
    n = len(layers)
    sim = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            v1 = layers[i][0, token_idx].cpu()
            v2 = layers[j][0, token_idx].cpu()
            sim[i, j] = F.cosine_similarity(v1, v2, dim=0).item()
    return sim


# --------------------------------------------------------------------------- #
#  Plot util (unchanged except for minor refactor)
# --------------------------------------------------------------------------- #
def plot_upper_triangle(
    sim_matrix, title, save_path, cmap="Greens",
    vmin=0.0, vcenter=0.5, vmax=1.0,
    lower_color="lightgray", grid_color="white", grid_linewidth=0.25
):
    n = sim_matrix.shape[0]
    upper_mask = np.tril(np.ones_like(sim_matrix, dtype=bool), k=-1)
    upper = np.ma.masked_array(sim_matrix, mask=upper_mask)

    # lower-triangle background
    bg = np.full_like(sim_matrix, np.nan, dtype=np.float32)
    bg[upper_mask] = 0.5
    bg = np.ma.masked_invalid(bg)

    plt.figure(figsize=(8, 7))
    plt.imshow(bg, cmap="Greys", vmin=0, vmax=1)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = plt.imshow(upper, cmap=cmap, norm=norm)
    plt.colorbar(im, label="Cosine similarity")
    plt.title(title)
    plt.xlabel("Layer j"); plt.ylabel("Layer i")

    plt.gca().set_xticks(np.arange(-.5, n, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, n, 1), minor=True)
    plt.grid(True, which="minor", color=grid_color, linewidth=grid_linewidth)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------- #
#  Prompt / input builder  (same as before)
# --------------------------------------------------------------------------- #
def build_prompt_and_inputs(
    instruction, image_path, tokenizer, image_transform, prompt_builder_fn
):
    prompt_builder = prompt_builder_fn("openvla")
    prompt_builder.add_turn(
        "human", f"What action should the robot take to {instruction.lower()}?"
    )
    prompt = prompt_builder.get_prompt()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = pad_sequence(
        input_ids, batch_first=True,
        padding_value=tokenizer.pad_token_id
    )[:, : tokenizer.model_max_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    img = Image.open(image_path).convert("RGB")
    pixel_vals = image_transform(img)
    pixel_vals = {k: v.unsqueeze(0) for k, v in pixel_vals.items()}

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_vals
    )


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VLA or VLM model
    VLA_SIM = True
    model = None
    if VLA_SIM:
        vla_path = Path("/home/apilaka/edgevla/checkpoints/vla/llava-lvis-lrv-openx/checkpoints/latest-checkpoint.pt")
        vla = load_vla(vla_path, hf_token=os.environ["HF_TOKEN"]).to(device).eval()
        model = vla
    else:
        vlm_path = Path("/home/apilaka/edgevla/checkpoints/vlm/llava-lvis-lrv")
        vlm = load(vlm_path, hf_token=os.environ["HF_TOKEN"]).to(device).eval()
        model = vlm

    tokenizer = model.llm_backbone.get_tokenizer()
    img_tx = model.vision_backbone.get_image_transform()

    num_samples = 350
    mode = args.mode.lower()             # 'prefix' or 'action'
    agg_type = 'avg'                     # or 'min'

    agg_sim = None                       # will init after first pass
    print(f"Running in **{mode}** mode")

    with open(
        "/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/"
        "sample_instructions.txt"
    ) as f:
        for idx, line in enumerate(f, start=1):
            if idx > num_samples: break
            instr = line.strip()
            img_path = (
                f"/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/"
                f"imgs/sample_img{idx}.png"
            )
            inputs = build_prompt_and_inputs(
                instr, img_path, tokenizer, img_tx,
                model.llm_backbone.prompt_builder_fn
            )
            # move to device
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            inputs["pixel_values"] = {k: v.to(device)
                                      for k, v in inputs["pixel_values"].items()}

            print(f"[{idx}/{num_samples}] computing…", end="\r")

            if mode == "prefix":
                sim = cosine_matrix_single_token(
                    model(**inputs, output_hidden_states=True).hidden_states,
                    token_idx=random.randint(
                        0, inputs["input_ids"].shape[1]-1
                    )
                )

            else:  # --- ACTION-TOKEN MODE ---------------------------------- #
                past = None
                ip_ids = inputs["input_ids"]
                attn   = inputs["attention_mask"]
                pix    = inputs["pixel_values"]
                L = model.llm_backbone.llm.config.num_hidden_layers
                sim_acc = np.zeros((L, L), dtype=np.float32)

                for step in range(7):
                    print(f"  Generating action token {step+1}/7", end="\r")
                    autocast_dtype = model.llm_backbone.half_precision_dtype
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
                        out = model(
                            pixel_values=pix if step == 0 else None,
                            input_ids=ip_ids,
                            attention_mask=attn,
                            past_key_values=past,
                            use_cache=True,
                            output_hidden_states=True
                        )
                    sim_acc += cosine_matrix_single_token(out.hidden_states,
                                                          token_idx=-1)
                    print(f"Output Logits Shape: {out.logits.shape}")
                    # next token (greedy)
                    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1,
                                            keepdim=True)
                    print(f"Predicted Token ID: {next_tok.item()}")
                    ip_ids = next_tok          # only the newly generated token
                    attn   = torch.ones_like(next_tok)
                    past   = out.past_key_values
                    pix    = None              # only need image first step

                sim = sim_acc / 7.0           # mean over 7 tokens
                exit()
            # -------- Agglomerate across samples -------------------------- #
            if agg_sim is None:
                agg_sim = (np.full_like(sim, 1.0) if agg_type == 'min'
                           else np.zeros_like(sim))
            agg_sim = (np.minimum(agg_sim, sim) if agg_type == 'min'
                       else agg_sim + sim / num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(Path(args.output_dir) / f"cosine_{mode}_pre.npy" if not VLA_SIM else Path(args.output_dir) / f"cosine_{mode}_post.npy", agg_sim)
    plot_upper_triangle(
        agg_sim,
        f"{mode.capitalize()} Cosine Similarity",
        Path(args.output_dir) / f"cosine_{mode}_pre.png" if not VLA_SIM else Path(args.output_dir) / f"cosine_{mode}_post.png",
    )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./cosine_similarity_results_robotics")
    p.add_argument("--mode", choices=["prefix", "action"], default="prefix",
                   help="'prefix' for instruction-grounding, "
                        "'action' for 7-token action analysis")
    main(p.parse_args())
