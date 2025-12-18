import os, re, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

UNUSED_CACHE_KEYS = {
    "input_video",
    "noise"
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
        resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
        allow_incomplete_state = getattr(args, "resume_allow_incomplete_state", False)
        dataloader_seed = getattr(args, "dataloader_seed", None)
    else:
        resume_from_checkpoint = None
        allow_incomplete_state = False
        dataloader_seed = None

    # Prepare dataloader RNG/state for deterministic resume
    resume_metadata = {}
    if resume_from_checkpoint is not None:
        resume_metadata = model_logger.read_checkpoint_metadata(resume_from_checkpoint)

    # Generate / synchronize dataloader seed across ranks
    if dataloader_seed is None:
        dataloader_seed = resume_metadata.get("dataloader_seed", torch.seed())
    dataloader_seed_tensor = torch.tensor([int(dataloader_seed)], device="cpu", dtype=torch.int64)
    dataloader_seed = int(accelerator.broadcast(dataloader_seed_tensor)[0].item())
    
    dataloader_generator = torch.Generator(device="cpu")
    if dataloader_seed is not None:
        dataloader_generator.manual_seed(int(dataloader_seed))

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    sampler = torch.utils.data.RandomSampler(dataset, generator=dataloader_generator)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Initialize wandb
    model_logger.init_wandb(accelerator)
    
    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    start_epoch = 0
    resume_step_in_epoch = -1
    if resume_from_checkpoint is not None:
        resume_metadata = model_logger.load_training_state(
            accelerator,
            resume_from_checkpoint,
            allow_incomplete_state=allow_incomplete_state,
        )
        start_epoch = int(resume_metadata.get("epoch", 0) or 0)
        resume_step_in_epoch = resume_metadata.get("step_in_epoch", -1)
        if resume_step_in_epoch is None:
            resume_step_in_epoch = -1
        else:
            resume_step_in_epoch = int(resume_step_in_epoch)
        if "steps_per_epoch" in resume_metadata:
            steps_per_epoch = int(resume_metadata["steps_per_epoch"])
        if model_logger.num_steps == 0 and resume_from_checkpoint is not None:
            ckpt_name = os.path.basename(resume_from_checkpoint.rstrip("/"))
            match = re.search(r"step-(\d+)", ckpt_name)
            if match:
                model_logger.num_steps = int(match.group(1))
        if steps_per_epoch is not None and resume_step_in_epoch >= steps_per_epoch - 1:
            start_epoch = min(num_epochs, start_epoch + 1)
            resume_step_in_epoch = -1
    
    last_trained_step_in_epoch = -1
    last_epoch = start_epoch - 1
    for epoch_id in range(start_epoch, num_epochs):
        last_epoch = epoch_id
        pbar = tqdm(dataloader, ncols=80, dynamic_ncols=False)
        for step_in_epoch, data in enumerate(pbar):
            if epoch_id == start_epoch and resume_step_in_epoch >= 0 and step_in_epoch <= resume_step_in_epoch:
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    save_steps,
                    loss=loss,
                    epoch_id=epoch_id,
                    step_in_epoch=step_in_epoch,
                    steps_per_epoch=steps_per_epoch,
                    dataloader_seed=dataloader_seed,
                )
                scheduler.step()
                last_trained_step_in_epoch = step_in_epoch
                # show loss on tqdm (no try/except as requested)
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                pbar.set_postfix({"loss": f"{loss_val:.6f}"})
        if save_steps is None:
            model_logger.on_epoch_end(
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch_id,
                steps_per_epoch,
                dataloader_seed=dataloader_seed,
            )
    model_logger.on_training_end(
        accelerator,
        model,
        optimizer,
        scheduler,
        save_steps,
        last_epoch,
        last_trained_step_in_epoch,
        steps_per_epoch,
        dataloader_seed=dataloader_seed,
    )


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
    
    for data_id, data in enumerate(tqdm(dataloader)):
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
