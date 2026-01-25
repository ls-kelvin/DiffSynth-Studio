#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
from typing import Iterable, Tuple

import torch
from tqdm import tqdm

from diffsynth.pipelines.wan_video_inter import WanVideoUnit_BlockScheduler


def iter_pth_files(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    if name.endswith(".pth"):
                        yield os.path.join(root, name)
        elif path.endswith(".pth"):
            yield path


def cap_clip_frames(
    prompt_list: list,
    clip_frames: list,
    max_frames: int,
) -> Tuple[list, list]:
    if max_frames <= 0 or not clip_frames:
        return [], []
    new_clip_frames = []
    total = 0
    for frames in clip_frames:
        if total + frames > max_frames:
            break
        new_clip_frames.append(int(frames))
        total += int(frames)
    new_prompt_list = prompt_list[:len(new_clip_frames)]
    return new_prompt_list, new_clip_frames


def trim_latents(input_latents, num_latent_frames: int):
    if not torch.is_tensor(input_latents):
        return input_latents
    if input_latents.dim() == 5:
        return input_latents[:, :, :num_latent_frames, ...]
    if input_latents.dim() == 4:
        return input_latents[:, :num_latent_frames, ...]
    if input_latents.dim() == 3:
        return input_latents[:num_latent_frames, ...]
    return input_latents


def slice_token_axis(tensor, new_len: int):
    if not torch.is_tensor(tensor):
        return tensor
    if tensor.dim() == 2:
        return tensor[:, :new_len]
    if tensor.dim() == 3:
        return tensor[:, :new_len, :]
    return tensor


def process_cached_data(data, max_frames: int):
    if isinstance(data, (list, tuple)) and len(data) == 3:
        inputs_shared, inputs_posi, inputs_nega = data
    elif isinstance(data, dict):
        inputs_shared, inputs_posi, inputs_nega = data, {}, {}
    else:
        raise ValueError("Unsupported cached data format; expected tuple/list of len 3 or dict.")

    clip_frames = inputs_shared.get("clip_frames")
    prompt_list = inputs_shared.get("prompt_list")
    if clip_frames is None or prompt_list is None:
        raise ValueError("Missing clip_frames or prompt_list in cached data.")

    new_prompt_list, new_clip_frames = cap_clip_frames(prompt_list, clip_frames, max_frames)
    if not new_clip_frames:
        raise ValueError("No clips remain after applying num_frames cap.")

    new_num_frames = int(sum(new_clip_frames))
    original_num_frames = int(inputs_shared.get("num_frames", new_num_frames))

    if new_num_frames >= original_num_frames:
        return data, False

    block_info = WanVideoUnit_BlockScheduler().process(
        None,
        new_prompt_list,
        new_clip_frames,
        new_num_frames,
    )["block_info"]

    inputs_shared["prompt_list"] = new_prompt_list
    inputs_shared["clip_frames"] = new_clip_frames
    inputs_shared["block_info"] = block_info
    inputs_shared["num_frames"] = new_num_frames

    if "prompt_embeddings_map" in inputs_shared:
        max_prompt_idx = len(new_prompt_list)
        inputs_shared["prompt_embeddings_map"] = {
            k: v for k, v in inputs_shared["prompt_embeddings_map"].items()
            if int(k) < max_prompt_idx
        }

    num_latent_frames = 1 + (new_num_frames - 1) // 4
    if "input_latents" in inputs_shared:
        inputs_shared["input_latents"] = trim_latents(
            inputs_shared["input_latents"], num_latent_frames
        )

    mllm_mask = inputs_shared.get("mllm_mask")
    mllm_vision_ranges = inputs_shared.get("mllm_vision_ranges")
    if mllm_mask is not None or mllm_vision_ranges is not None:
        height = inputs_shared.get("height")
        width = inputs_shared.get("width")
        lat_h = lat_w = None
        if isinstance(height, int) and isinstance(width, int):
            lat_h = height // 16
            lat_w = width // 16
        elif torch.is_tensor(inputs_shared.get("input_latents")):
            lat_h = int(inputs_shared["input_latents"].shape[-2])
            lat_w = int(inputs_shared["input_latents"].shape[-1])
        if lat_h is not None and lat_w is not None:
            tokens_per_latent_frame = lat_h * lat_w
            new_token_len = num_latent_frames * tokens_per_latent_frame
            if mllm_mask is not None:
                inputs_shared["mllm_mask"] = slice_token_axis(mllm_mask, new_token_len)
            if mllm_vision_ranges is not None:
                inputs_shared["mllm_vision_ranges"] = slice_token_axis(mllm_vision_ranges, new_token_len)

    if isinstance(data, tuple):
        return (inputs_shared, inputs_posi, inputs_nega), True
    if isinstance(data, list):
        return [inputs_shared, inputs_posi, inputs_nega], True
    return inputs_shared, True


def _process_file(in_path: str, out_path: str, max_frames: int, dry_run: bool):
    try:
        data = torch.load(in_path, map_location="cpu", weights_only=False)
        new_data, _ = process_cached_data(data, max_frames)
        if not dry_run:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(new_data, out_path)
        return in_path, out_path, None
    except Exception as exc:
        return in_path, out_path, str(exc)


def _process_file_args(args_tuple):
    return _process_file(*args_tuple)


def main():
    parser = argparse.ArgumentParser(
        description="Trim cached Wan MLLM .pth files to a max num_frames without re-encoding."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Input .pth files or directories containing cached data.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Maximum number of frames to keep (tail is dropped).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trimmed .pth files. Defaults to in-place if not set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without writing output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count).",
    )
    args = parser.parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")

    abs_inputs = [os.path.abspath(p) for p in args.paths]
    base_dir = os.path.commonpath(abs_inputs)

    inputs = []
    for in_path in iter_pth_files(args.paths):
        out_path = in_path
        if args.output_dir is not None:
            rel = os.path.relpath(os.path.abspath(in_path), start=base_dir)
            out_path = os.path.join(args.output_dir, rel)
        inputs.append((in_path, out_path))

    if not inputs:
        print("No .pth files found.")
        return

    worker_count = args.workers or os.cpu_count() or 1
    worker_count = max(1, int(worker_count))
    with mp.Pool(processes=worker_count) as pool:
        iterator = pool.imap_unordered(
            _process_file_args,
            [(p, o, args.num_frames, args.dry_run) for p, o in inputs],
        )
        for in_path, out_path, err in tqdm(iterator, total=len(inputs), ncols=120):
            if err:
                print(f"[skip] {in_path}: {err}")
                continue
            # print(f"[write] {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
