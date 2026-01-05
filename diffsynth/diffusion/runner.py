import os, json, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

UNUSED_CACHE_KEYS = {
    "input_video",
    "noise",
    "latents"
}


def prune_cached_data(data):
    """Remove raw media fields that are not needed for stage-2 split training."""
    if isinstance(data, dict):
        for key in list(data.keys()):
            if key in UNUSED_CACHE_KEYS:
                data.pop(key, None)
            else:
                data[key] = prune_cached_data(data[key])
    elif isinstance(data, (list, tuple)):
        data = type(data)(prune_cached_data(x) for x in data)
    return data


def _get_dataloader_seed(args, accelerator):
    """Get or generate a seed for dataloader shuffling."""
    import random
    if args is not None and hasattr(args, 'dataloader_seed') and args.dataloader_seed is not None:
        return args.dataloader_seed
    # Generate a random seed and sync across all processes
    if accelerator.is_main_process:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = 0
    seed = accelerator.gather(torch.tensor([seed], device=accelerator.device))[0].item()
    return int(seed)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    # Get resume parameters
    resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None) if args is not None else None
    resume_allow_incomplete_state = getattr(args, 'resume_allow_incomplete_state', False) if args is not None else False
    
    # Get dataloader seed for reproducibility
    dataloader_seed = _get_dataloader_seed(args, accelerator)
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers, generator=generator
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Resume from checkpoint if specified
    starting_epoch = 0
    resume_step = 0
    if resume_from_checkpoint is not None:
        if os.path.isdir(resume_from_checkpoint):
            accelerator.print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            
            # Load training metadata
            metadata_path = os.path.join(resume_from_checkpoint, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                starting_epoch = metadata.get("epoch", 0)
                resume_step = metadata.get("global_step", 0)
                model_logger.num_steps = resume_step
                accelerator.print(f"Resuming from epoch {starting_epoch}, step {resume_step}")
        else:
            accelerator.print(f"Warning: Checkpoint path {resume_from_checkpoint} does not exist, starting from scratch")
    
    # Store dataloader seed in model_logger for checkpoint saving
    model_logger.dataloader_seed = dataloader_seed
    
    # Initialize wandb
    model_logger.init_wandb(accelerator)
    
    # Skip dataloader to the correct position if resuming
    skip_steps = resume_step - starting_epoch * len(dataloader)

    for epoch_id in range(starting_epoch, num_epochs):
        dataloader.set_epoch(epoch_id)

        if epoch_id == starting_epoch:
            dataloader_iter = accelerator.skip_first_batches(dataloader, num_batches=skip_steps)
        else:
            dataloader_iter = dataloader

        pbar = tqdm(dataloader_iter, ncols=130, dynamic_ncols=False, 
                   desc=f"Epoch {epoch_id + 1}/{num_epochs}")
        accumulated_loss = torch.tensor(0.0, device=accelerator.device)
        micro_steps = 0

        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                loss_detached = loss.detach()
                accumulated_loss += loss_detached
                micro_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                # Only log to wandb once per full (global) batch, but keep num_steps unchanged
                if accelerator.sync_gradients:
                    total_loss = accelerator.gather(accumulated_loss).mean()
                    global_loss = total_loss / micro_steps
                    log_loss = global_loss
                    loss_to_report = global_loss
                    accumulated_loss.zero_()
                    micro_steps = 0
                else:
                    log_loss = None
                    loss_to_report = loss_detached
                model_logger.on_step_end(accelerator, model, save_steps, loss=log_loss, epoch=epoch_id)
                loss_val = loss_to_report.item() if isinstance(loss_to_report, torch.Tensor) else float(loss_to_report)
                pbar.set_postfix({"loss": f"{loss_val:.6f}"})
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    os.makedirs(model_logger.output_path, exist_ok=True)
    
    for data_id, data in enumerate(tqdm(dataloader, ncols=130, dynamic_ncols=False)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                # folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                # os.makedirs(folder, exist_ok=True)
                # save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                save_path = os.path.join(model_logger.output_path, f"{data['video_id']}.pth")
                if os.path.exists(save_path):
                    continue
                data = model(data)
                data = prune_cached_data(data)
                torch.save(data, save_path)
