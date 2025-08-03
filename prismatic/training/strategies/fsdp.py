"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.amp import GradScaler
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class FSDPStrategy(TrainingStrategy):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sharding_strategy: str = "shard-grad-op",
        using_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            using_lora=using_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP Sharding Strategy {sharding_strategy} is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        # === LoRA MODIFICATION START ===
        # Extract and save LoRA weights separately

        if hasattr(self, 'using_lora') and self.using_lora:
            from peft.utils import get_peft_model_state_dict
            with FSDP.summon_full_params(self.vlm):
                # Unwrap model properly
                unwrapped_model = self.vlm.module if hasattr(self.vlm, 'module') else self.vlm
                # should now be a PEFTModel
                if overwatch.is_rank_zero():
                    print("Attempting to save LoRA checkpoint")
                    
                    # Use `safe_save=True` to avoid NCCL issues
                    lora_state_dict = get_peft_model_state_dict(unwrapped_model)
                    print("Preparing checkpoint")

                    # Create LoRA-specific checkpoint path
                    if train_loss is None:
                        lora_checkpoint_dir = run_dir / f"lora-step-{global_step:06d}-epoch-{epoch:02d}-loss-inf"
                    else:
                        lora_checkpoint_dir = run_dir / f"lora-step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}"
                    lora_checkpoint_dir.mkdir(exist_ok=True, parents=True)

                    print("Saving LoRA checkpoint")
                    # torch.save(lora_state_dict, lora_checkpoint_path)
                    unwrapped_model.save_pretrained(lora_checkpoint_dir)
                    
                    # # Save additional training state
                    # checkpoint = {
                    #     'epoch': epoch,
                    #     'step': step,
                    #     'optimizer': optimizer.state_dict() if optimizer else None,
                    #     'scheduler': scheduler.state_dict() if scheduler else None,
                    # }
                    # torch.save(checkpoint, os.path.join(lora_checkpoint_path, "training_state.pt"))

                    # # Create symlink to latest checkpoint (fix symlink path issue)
                    # latest_path = lora_checkpoint_dir / "latest-lora-checkpoint.pt"
                    # if latest_path.exists():
                    #     latest_path.unlink()
                    # latest_path.symlink_to(lora_checkpoint_path.resolve())

                    # print(f"Saved LoRA checkpoint to {lora_checkpoint_path}")
        # === LoRA MODIFICATION END ===
        else:
            # Summon Full State Dictionary =>> Reconstitute from Shards
            with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
                full_vlm_state_dict = self.vlm.state_dict()
                model_state_dicts = {
                    mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
                }

                # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
                for key, param in full_vlm_state_dict.items():
                    for mkey in model_state_dicts:
                        if key.startswith(mprefix := f"{mkey}."):
                            model_state_dicts[mkey][key.removeprefix(mprefix)] = param

                # Save on rank zero *only*
                if overwatch.is_rank_zero():
                    checkpoint_dir = run_dir / "checkpoints"
                    if train_loss is None:
                        checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
                    else:
                        checkpoint_path = (
                            checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                        )

                    # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                    torch.save({"model": model_state_dicts}, checkpoint_path)

                    # TODO (siddk) :: This breaks w/ Sagemaker default permissions (root vs. <user>)... skip?
                    # shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
        dist.barrier()


    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()
        # # Assemble the Default FSDP Mixed Precision Policy
        # if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
        #     # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
        #     #   => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
        #     reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
        #     fsdp_precision_policy = MixedPrecision(
        #         param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype
        #     )

        #     # When running FSDP with a frozen vision backbone --> move to half precision!
        #     if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
        #         overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
        #         self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)

        # else:
        #     # If we're not using mixed precision, everything is in default full precision!
        #     fsdp_precision_policy = MixedPrecision(
        #         param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        #     )

        if self.enable_mixed_precision_training:
            if self.mixed_precision_dtype == torch.bfloat16:
                fsdp_precision_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16 if not self.reduce_in_full_precision else torch.float32,
                    buffer_dtype=torch.bfloat16,
                )
            else:  # FP16 path
                fsdp_precision_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16 if not self.reduce_in_full_precision else torch.float32,
                    buffer_dtype=torch.float16,
                )
            # When running FSDP with a frozen vision backbone --> move to half precision!
            if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
                overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )


        # Initialize distributed process group
        # dist.init_process_group(backend="nccl", world_size=overwatch.world_size())
        # local_rank = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(local_rank)
        # corda_config = CordaConfig(
        #     corda_method="kpm",
        # )

        if hasattr(self, 'using_lora') and self.using_lora:
            # Load model and apply LoRA
            lora_config = LoraConfig(
                r=self.lora_rank,                      # Modest rank for decent adaptation
                lora_alpha=self.lora_alpha,            # 2x the rank value for moderate scaling
                target_modules=[
                    # For TinyLlama language model
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
                    "gate_proj", "up_proj", "down_proj",     # MLP modules
                                                            # For connecting DINOv2 vision backbone
                    "mm_projector",                         # Vision-to-language projection
                ],
                lora_dropout=0.05,        # Small dropout for regularization
                bias="none",              # No bias adaptation to save parameters
                task_type="CAUSAL_LM",     # Since TinyLlama is causal
                init_lora_weights="gaussian",  # Initialize LoRA weights with Gaussian distribution
                # corda_config=corda_config
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            # self.vlm.print_trainable_parameters()

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vlm = FSDP(
            self.vlm,
            auto_wrap_policy=vlm_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, betas=(0.9, 0.97), lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, betas=(0.9, 0.97), lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")
        
        self.scaler = GradScaler(
            device="cuda", 
            enabled=self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.float16, 
            init_scale=2**10,
            )
        trainable_params = 1
        total_params = 1
        if not self.using_lora:
            trainable_params = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vlm.parameters())

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"                 |-> Using Loss Scaling? = {self.scaler.is_enabled()}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> Training Epochs = {self.epochs}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
            f"         |-> Using LoRA? = {self.using_lora}\n"
            f"         |-> Trainable params: {self.vlm.module.print_trainable_parameters() if self.using_lora else trainable_params} ({100*trainable_params/total_params}%)\n"
        )

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        self.vlm.clip_grad_norm_(max_norm=self.max_grad_norm)
