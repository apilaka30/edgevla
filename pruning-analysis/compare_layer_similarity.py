import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import argparse
import os
from pathlib import Path

from PIL import Image
from prismatic.models import load

from torch.nn.utils.rnn import pad_sequence
import random
from CKA import linear_CKA

random.seed(42)  # For reproducibility

@torch.no_grad()
def get_llm_block_outputs(model, pixel_values, input_ids, attention_mask):
    outputs = []
    # print(f"input_ids dimension: {input_ids.shape}\nPixel_Values: {(pixel_values['dino'].shape, pixel_values['siglip'].shape)}\nNum patches: {model.vision_backbone.num_patches}")
    # print(f"Attention mask: {attention_mask.shape}")

    # Forward pass through the full VLM model
    autocast_dtype = model.llm_backbone.half_precision_dtype
    model_output = None
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
        model_output = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    outputs = model_output.hidden_states

    return outputs


def compute_cosine_matrix(outputs):
    n = len(outputs)
    # print(f"Hidden state dimension tensors: {outputs[0].shape}")
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            token_idx = random.randint(0, outputs[i].shape[1]-1)  # Randomly select a token index
            v1 = outputs[i][0][token_idx].cpu()
            v2 = outputs[j][0][token_idx].cpu()
            sim_matrix[i, j] = linear_CKA(X=v1.unsqueeze(0), Y=v2.unsqueeze(0))#F.cosine_similarity(v1, v2, dim=0).item()
    return sim_matrix


def plot_upper_triangle_similarity_matrix(
    sim_matrix,
    title,
    save_path,
    cmap="PiYG_r",
    vmin=0.0,
    vcenter=0.5,
    vmax=1.0,
    lower_color="lightgray",
    grid_color="white",
    grid_linewidth=0.25,
    show_values=False
):
    """
    Plot only the upper triangle of a similarity matrix using the PiYG colormap,
    and fill the lower triangle with a gray background like in the SLEB figure.
    """
    n = sim_matrix.shape[0]

    # Mask lower triangle (i > j)
    upper_mask = np.tril(np.ones_like(sim_matrix, dtype=bool), k=-1)
    upper_data = np.ma.masked_array(sim_matrix, mask=upper_mask)

    # Create gray background mask for lower triangle
    background = np.full_like(sim_matrix, fill_value=np.nan, dtype=np.float32)
    background[upper_mask] = 0.5  # middle value (centered gray in grayscale map)
    background = np.ma.masked_invalid(background)

    # Plot setup
    plt.figure(figsize=(8, 7))

    # Gray lower triangle
    plt.imshow(
        background,
        cmap=plt.cm.Greys,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    # Colored upper triangle with PiYG diverging colormap
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = plt.imshow(
        upper_data,
        cmap=cmap,
        # norm=norm,
        interpolation="nearest"
    )

    plt.colorbar(im, label="Cosine similarity")
    plt.title(title)
    plt.xlabel("Layer j Index")
    plt.ylabel("Layer i Index")

     # Grid
    # plt.xticks(np.arange(n))
    # plt.yticks(np.arange(n))
    # plt.grid(which="both", color=grid_color, linestyle="-", linewidth=grid_linewidth)
    plt.gca().set_xticks(np.arange(-0.5, n, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, n, 1), minor=True)
    plt.grid(True, which="minor", color=grid_color, linewidth=grid_linewidth)
    plt.tick_params(axis='both', which='major', labelsize=8)

    # Labels and layout
    plt.title(title)
    plt.xlabel("Layer j Index")
    plt.ylabel("Layer i Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_prompt_and_inputs(instruction, image_path, base_tokenizer, image_transform, prompt_builder_fn):
    """Mimics RLDSBatchTransform for a single image + instruction pair."""

    # Language prompt builder
    prompt_builder = prompt_builder_fn("openvla")
    lang = instruction.lower()
    prompt_builder.add_turn("human", f"What action should the robot take to {lang}?")

    prompt = prompt_builder.get_prompt()
    input_ids = base_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=base_tokenizer.pad_token_id)[:, : base_tokenizer.model_max_length]

    attention_mask = input_ids.ne(base_tokenizer.pad_token_id)

    # Image processing
    img = Image.open(image_path).convert("RGB")
    pixel_values = image_transform(img)
    pixel_values['dino'] = pixel_values['dino'].unsqueeze(0)  # add batch dimension
    pixel_values['siglip'] = pixel_values['siglip'].unsqueeze(0)  # add batch dimension
    # print("Input IDs shape:", input_ids.shape)
    # print(pixel_values['dino'].shape, pixel_values['siglip'].shape)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_path = Path("/home/apilaka/edgevla/checkpoints/vlm/llava-lvis-lrv")
    vlm = load(vlm_path, hf_token=os.environ["HF_TOKEN"], load_for_training=False).to(device).eval()
    tokenizer = vlm.llm_backbone.get_tokenizer()
    image_transform = vlm.vision_backbone.get_image_transform()
    # model_post = load_vla(vla_path, hf_token=os.environ["HF_TOKEN"], load_for_training=False).eval()

    agg_type = 'avg'  # ('min' or 'avg')
    agg_sim_matrix = np.full((23, 23), fill_value=1.0) if agg_type == 'min' else np.zeros((23, 23))
    num_samples = 120
    with open("/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/sample_instructions.txt", "r") as f:
        for i, line in enumerate(f, start=1):
            if i > num_samples:
                break
            text_prompt = line.strip()
            image_path = f"/home/apilaka/edgevla/openvla/pruning-analysis/robotics_samples/imgs/sample_img{i}.png"
            inputs = build_prompt_and_inputs(
                instruction=text_prompt,
                image_path=image_path,
                base_tokenizer=tokenizer,
                image_transform=image_transform,
                prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn
            )
            # Move inputs to device
            for k in inputs:
                if k == "pixel_values":
                    inputs[k] = {key: v.to(device) for key, v in inputs[k].items()}
                else:
                    inputs[k] = inputs[k].to(device)
            

            # Run pre-finetuning
            print(f"Computing pre-finetuning similarity...[{i}/{num_samples}]")
            out_pre = get_llm_block_outputs(vlm, **inputs)
            sim_pre = compute_cosine_matrix(out_pre)
            agg_sim_matrix = np.minimum(agg_sim_matrix, sim_pre) if agg_type == 'min' else agg_sim_matrix + sim_pre*(1/num_samples)
    np.save(os.path.join(args.output_dir, "cosine_pre.npy"), agg_sim_matrix)
    plot_upper_triangle_similarity_matrix(agg_sim_matrix, "Pre-Finetuning Cosine Similarity", os.path.join(args.output_dir, "cosine_pre.png"))

    # # Run post-finetuning
    # print("Computing post-finetuning similarity...")
    # out_post = get_llm_block_outputs(vla, **inputs)
    # sim_post = compute_cosine_matrix(out_post)
    # np.save(os.path.join(args.output_dir, "cosine_post.npy"), sim_post)
    # plot_similarity_matrix(sim_post, "Post-Finetuning Cosine Similarity", os.path.join(args.output_dir, "cosine_post.png"))

    # Difference
    # delta = sim_post - sim_pre
    # np.save(os.path.join(args.output_dir, "cosine_delta.npy"), delta)
    # plot_similarity_matrix(delta, "Post - Pre Similarity Δ", os.path.join(args.output_dir, "cosine_delta.png"))

    # print("Saved results to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/apilaka/edgevla/openvla/pruning-analysis/robotics_sample/sample_image.png",
                        help="Path to input image")
    parser.add_argument("--text_prompt", type=str, default="What",
                        help="Language instruction to build the prompt")
    parser.add_argument("--output_dir", type=str, default="./cosine_similarity_results_robotics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
