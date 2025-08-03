import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
from collections import OrderedDict
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

from pathlib import Path
from typing import Callable, Optional

from peft import PeftModel

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    # vla: VLAConfig = field(
    #     default_factory=VLAConfig.get_choice_class(VLARegistry.EDGEVLA.vla_id)
    # )
    # vla: VLAConfig = field(
    #     default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    # )
    

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/bigscratch/apilaka/rlds_datasets/open_x_embodiment"
    )
    run_root_dir: Path = Path("/bigscratch/apilaka/vla/runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = False                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 1000                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 42                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl",)                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

    # fmt: on

@draccus.wrap()
def load_and_test(cfg: TrainConfig) -> None:
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    vlm = load_vla(Path("/home/apilaka/edgevla/checkpoints/vla/llava-lrv-openx/checkpoints/step-3000-epoch-00-loss=3.0724.pt"), hf_token=hf_token, load_for_training=True)

    # # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    # overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones("vla-train")

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Loading LoRA Adapter Weights from /bigscratch/apilaka/vla/runs/edgevla+n1+b108+x7/lora-step-006000-epoch-00-loss=2.9216")
    # vlm = PrismaticVLM.from_pretrained(cfg.pretrained_checkpoint, cfg.model.model_id, vision_backbone=vision_backbone, llm_backbone=llm_backbone, arch_specifier=cfg.model.arch_specifier)
    vlm = PeftModel.from_pretrained(vlm,  Path("/bigscratch/apilaka/vla/runs/edgevla+n1+b108+x7/lora-step-006000-epoch-00-loss=2.9216"))
    vlm = vlm.merge_and_unload()
    
    full_vlm_state_dict = vlm.state_dict()
    model_state_dicts = {
        mkey: OrderedDict() for mkey in vlm.all_module_keys
    }

#     # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
    for key, param in full_vlm_state_dict.items():
        for mkey in model_state_dicts:
            if key.startswith(mprefix := f"{mkey}."):
                model_state_dicts[mkey][key.removeprefix(mprefix)] = param

    checkpoint_path = "/home/apilaka/edgevla/checkpoints/vla/llava-lrv-openx/checkpoints/step-6000-epoch-00-loss=2.9216.pt"
    overwatch.info(f"Saving Converted VLA Checkpoint to /home/apilaka/edgevla/checkpoints/vla/llava-lrv-openx/checkpoints/step-6000-epoch-00-loss=2.9216.pt")
    # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
    torch.save({"model": model_state_dicts}, checkpoint_path)


    # # SAVE THIS VLM CHECKPOINT

    # overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    # train_dataset, collator = get_dataset_and_collator(
    #     cfg.stage,
    #     cfg.dataset,
    #     image_transform,
    #     tokenizer,
    #     prompt_builder_fn=llm_backbone.prompt_builder_fn,
    #     default_image_resolution=vision_backbone.default_image_resolution,
    #     padding_side=tokenizer.padding_side,
    # )

    # grad_accumulation_steps = cfg.global_batch_size // cfg.per_device_batch_size // overwatch.world_size()
    # run_dir = cfg.run_root_dir / cfg.run_id
    # print(run_dir)
    # overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    # metrics = Metrics(
    #     cfg.trackers,
    #     cfg.run_id,
    #     run_dir,
    #     draccus.encode(cfg),
    #     cfg.stage,
    #     grad_accumulation_steps=grad_accumulation_steps,
    # )
    # run_evaluation(vlm, train_dataset, collator, metrics, cfg.global_batch_size, cfg.per_device_batch_size, worker_init_fn)


def run_evaluation(
        self,
        vlm,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        global_batch_size: int,
        per_device_batch_size: int,
        worker_init_fn,
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the evaluation loop for the given `dataset` and `collator`; log results to `metrics`"""
        
        # Use the same sampler as in training but without shuffling
        if batch_construction_strategy == "split-modality":
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=False,  # No shuffling for evaluation
                seed=seed,
                drop_last=False,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )

        # === Evaluate ===
        vlm.eval()
        status = metrics.get_status()
        with torch.no_grad():  # Ensure no gradients are computed
            with tqdm(
                total=len(dataloader),
                desc=status,
                leave=False,
                disable=not overwatch.is_rank_zero(),
            ) as progress:
                for eval_idx, batch in enumerate(dataloader):
                    with torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=True,
                    ):
                        output: CausalLMOutputWithPast = vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Evaluation Metrics
                    metrics.commit(loss=loss)
                    progress.update()
                    progress.set_description(status)
                
                # Push final status
                metrics.push()


if __name__ == "__main__":
    load_and_test()
