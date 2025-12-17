import os, torch
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
    
    def log_metrics(self, metrics: dict):
        """Log metrics to wandb"""
        if self.use_wandb and self.wandb_initialized:
            self.wandb.log(metrics, step=self.num_steps)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, loss=None):
        self.num_steps += 1
        
        # Log loss to wandb
        if loss is not None and accelerator.is_main_process:
            self.log_metrics({"train/loss": loss.item() if isinstance(loss, torch.Tensor) else loss})
        
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        
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
