import json, os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, use_wandb=False, wandb_project=None, wandb_run_name=None, wandb_config=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.use_wandb = use_wandb
        self.wandb_initialized = False
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_project = wandb_project
                self.wandb_run_name = wandb_run_name
                self.wandb_config = wandb_config or {}
            except ImportError:
                print("Warning: wandb is not installed. Logging will be disabled. Install it with: pip install wandb")
                self.use_wandb = False


    def init_wandb(self, accelerator: Accelerator):
        """Initialize wandb only on main process"""
        if self.use_wandb and not self.wandb_initialized and accelerator.is_main_process:
            self.wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.wandb_config,
            )
            self.wandb_initialized = True

    @staticmethod
    def _normalize_checkpoint_path(checkpoint_path: str):
        if checkpoint_path is None:
            return None
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".safetensors"):
            checkpoint_path = checkpoint_path.replace(".safetensors", "")
        return checkpoint_path

    def read_checkpoint_metadata(self, checkpoint_path: str):
        """Lightweight helper to read checkpoint meta.json without loading weights."""
        checkpoint_path = self._normalize_checkpoint_path(checkpoint_path)
        if checkpoint_path is None:
            return {}
        meta_path = os.path.join(checkpoint_path, "meta.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r") as f:
            return json.load(f)
    
    def log_metrics(self, metrics: dict):
        """Log metrics to wandb"""
        if self.use_wandb and self.wandb_initialized:
            self.wandb.log(metrics, step=self.num_steps)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer=None, scheduler=None, save_steps=None, loss=None, epoch_id=None, step_in_epoch=None, steps_per_epoch=None, dataloader_seed=None):
        self.num_steps += 1
        
        # Log loss to wandb
        if loss is not None and accelerator.is_main_process:
            self.log_metrics({"train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss})
        
        if save_steps is not None and self.num_steps % save_steps == 0:
            checkpoint_name = f"step-{self.num_steps}"
            self.save_checkpoint(
                accelerator,
                model,
                checkpoint_name,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata={"epoch": epoch_id, "step_in_epoch": step_in_epoch, "steps_per_epoch": steps_per_epoch, "dataloader_seed": dataloader_seed},
            )


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer=None, scheduler=None, epoch_id=None, steps_per_epoch=None, dataloader_seed=None):
        checkpoint_name = f"epoch-{epoch_id}"
        metadata = {
            "epoch": epoch_id,
            "step_in_epoch": (steps_per_epoch - 1) if steps_per_epoch is not None else None,
            "steps_per_epoch": steps_per_epoch,
            "dataloader_seed": dataloader_seed,
        }
        self.save_checkpoint(
            accelerator,
            model,
            checkpoint_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
        )


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, optimizer=None, scheduler=None, save_steps=None, epoch_id=None, step_in_epoch=None, steps_per_epoch=None, dataloader_seed=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            checkpoint_name = f"step-{self.num_steps}"
            self.save_checkpoint(
                accelerator,
                model,
                checkpoint_name,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata={"epoch": epoch_id, "step_in_epoch": step_in_epoch, "steps_per_epoch": steps_per_epoch, "dataloader_seed": dataloader_seed},
            )
        
        # Finish wandb run
        if self.use_wandb and self.wandb_initialized:
            self.wandb.finish()


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


    def save_training_state(self, accelerator: Accelerator, checkpoint_name: str, metadata=None):
        """Save optimizer/scheduler/rng states for resuming training."""
        checkpoint_dir = os.path.join(self.output_path, checkpoint_name)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            accelerator.save_state(checkpoint_dir, safe_serialization=True)
        except TypeError:
            accelerator.save_state(checkpoint_dir)
        if accelerator.is_main_process:
            metadata = {} if metadata is None else dict(metadata)
            metadata.setdefault("step", self.num_steps)
            metadata.setdefault("checkpoint_name", checkpoint_name)
            meta_path = os.path.join(checkpoint_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        accelerator.wait_for_everyone()
        return checkpoint_dir


    def save_checkpoint(self, accelerator: Accelerator, model: torch.nn.Module, checkpoint_name: str, optimizer=None, scheduler=None, metadata=None):
        """Save both lightweight trainable weights and full training state."""
        self.save_model(accelerator, model, f"{checkpoint_name}.safetensors")
        if optimizer is not None or scheduler is not None:
            self.save_training_state(accelerator, checkpoint_name, metadata=metadata)


    def load_training_state(self, accelerator: Accelerator, checkpoint_path: str, allow_incomplete_state: bool = False):
        """Load model/optimizer/scheduler/rng states for resuming training."""
        if checkpoint_path is None:
            return {}
        checkpoint_path = self._normalize_checkpoint_path(checkpoint_path)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
        accelerator.wait_for_everyone()
        try:
            accelerator.load_state(checkpoint_path, strict=not allow_incomplete_state)
        except TypeError:
            accelerator.load_state(checkpoint_path)

        metadata = self.read_checkpoint_metadata(checkpoint_path)
        gathered = accelerator.gather_object(metadata)
        metadata = next((m for m in gathered if m is not None), {})
        metadata = metadata or {}
        if "step" in metadata:
            self.num_steps = metadata["step"]
        return metadata
