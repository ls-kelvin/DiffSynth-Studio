import argparse
import sys
import traceback
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from diffsynth.core import UnifiedDataset
from diffsynth.core.data import InvalidDataError
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath


def build_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )


def validate_sample(sample, args, idx=None):
    sid = sample.get("video_id") or sample.get("id") or f"idx-{idx}"
    video = sample.get("video")
    if video is None or len(video) == 0:
        raise InvalidDataError(f"{sid}: empty video after preprocessing.")

    # Require at least one frame for MLLM video condition (indices start at frame 8)
    if len(video) < 9:
        raise InvalidDataError(f"{sid}: only {len(video)} frames; need >=9 to supply MLLM video context.")

    first_size = getattr(video[0], "size", None)
    if first_size is None:
        raise InvalidDataError(f"{sid}: first frame is not a valid image object.")

    for j, frame in enumerate(video):
        if frame is None or not hasattr(frame, "size"):
            raise InvalidDataError(f"{sid}: frame {j} is missing or invalid.")
        if frame.size != first_size:
            raise InvalidDataError(f"{sid}: frame {j} size {frame.size} differs from first frame {first_size}.")
        if args.check_nan:
            arr = np.array(frame, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise InvalidDataError(f"{sid}: non-finite pixel values in frame {j}.")

    factor, remainder = 4, 1  # Wan video expects T % 4 == 1
    n = len(video)
    if n % factor != remainder:
        raise InvalidDataError(f"{sid}: frame count {n} violates modulo constraint ({n} % {factor} != {remainder}).")

    prompt = sample.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        raise InvalidDataError(f"{sid}: prompt missing or empty.")


def main():
    parser = argparse.ArgumentParser(description="Offline MLLM dataset sanity check (CPU only).")
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--data_file_keys", type=str, default="video,prompt")
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_pixels", type=int, default=1920 * 1080)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on samples to scan.")
    parser.add_argument("--check_nan", action="store_true", help="Also check for NaN/inf pixels (slower).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for parallel validation.")
    args = parser.parse_args()

    ds = build_dataset(args)
    total = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    print(f"Scanning {total} / {len(ds)} samples...")

    num_bad = 0
    num_err = 0

    def process_idx(i):
        try:
            sample = ds[i]
            validate_sample(sample, args, idx=i)
            return ("ok", i, None)
        except InvalidDataError as e:
            return ("bad", i, str(e))
        except Exception as e:
            return ("error", i, f"{e}\n{traceback.format_exc()}")

    workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(total=total, ncols=80) as pbar:
        futures = [ex.submit(process_idx, i) for i in range(total)]
        for fut in as_completed(futures):
            status, idx, msg = fut.result()
            if status == "bad":
                num_bad += 1
                print(f"[BAD] {msg}")
            elif status == "error":
                num_err += 1
                print(f"[ERROR] idx {idx}: {msg}")
            pbar.update(1)

    print(f"Done. Invalid samples: {num_bad}, errors: {num_err}, valid samples: {total - num_bad - num_err}.")
    sys.exit(1 if (num_bad or num_err) else 0)


if __name__ == "__main__":
    main()
