import math
from typing import Iterable, List, Sequence, Tuple, Union

import torch
from PIL import Image


MLLM_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Analyze the provided context (either a text description for the first frame "
    "or a sequence of preceding video frames). Describe the key visual elements, "
    "ongoing motion, camera perspective, and overall scene composition. Then, "
    "based on the user's instruction, predict the visual content and dynamics "
    "of the next frame(s). Ensure the prediction maintains temporal coherence, "
    "logical progression, and consistency with the established style and action."
    "<|im_end|>\n<|im_start|>user\n"
)


def get_mllm_frame_indices(num_frames: int, mllm_fps: int = 2, base_fps: int = 16) -> List[int]:
    """Return frame indices to feed into the MLLM."""
    if num_frames <= 0:
        return []
    indices = [0]
    if base_fps <= 0 or mllm_fps <= 0:
        return indices
    frame_interval = max(1, base_fps // mllm_fps)
    for i in range(frame_interval, num_frames, base_fps):
        indices.extend([i, min(i + frame_interval, num_frames - 1)])
    return sorted(list(dict.fromkeys([i for i in indices if i < num_frames])))


def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    frame = frame.float()
    frame = frame.clamp(-1, 1).add(1).div(2).mul(255).byte()
    return Image.fromarray(frame.numpy())


def video_to_pil_frames(video: Union[Sequence[Image.Image], torch.Tensor]) -> List[Image.Image]:
    """Convert various video representations to a list of PIL frames."""
    if isinstance(video, (list, tuple)):
        if all(isinstance(frame, Image.Image) for frame in video):
            return list(video)
        if len(video) > 0 and isinstance(video[0], torch.Tensor):
            return [_tensor_to_pil(frame) for frame in video]
    if isinstance(video, torch.Tensor):
        if video.ndim == 5:
            # B C T H W or B T C H W
            if video.shape[0] == 1:
                video = video[0]
            else:
                video = video[0]
        if video.ndim == 4:
            # C T H W or T C H W
            if video.shape[0] in (1, 3):
                video = video.permute(1, 0, 2, 3)
        return [_tensor_to_pil(frame) for frame in video]
    raise TypeError(f"Unsupported video type: {type(video)}")


def sample_mllm_frames(
    video: Union[Sequence[Image.Image], torch.Tensor],
    num_frames: int,
    mllm_fps: int = 2,
    base_fps: int = 16,
) -> List[Image.Image]:
    """Sample frames according to the temporal layout expected by the MLLM."""
    frames = video_to_pil_frames(video)
    if len(frames) != num_frames:
        num_frames = len(frames)
    indices = get_mllm_frame_indices(num_frames, mllm_fps=mllm_fps, base_fps=base_fps)
    return [frames[i] for i in indices]


def normalize_pil_frames(
    frames: Iterable[Image.Image],
    size: Tuple[int, int] = None,
) -> List[torch.Tensor]:
    """Resize and normalize frames to [-1, 1] tensors."""
    processed = []
    for frame in frames:
        if size is not None:
            frame = frame.resize(size)
        tensor = torch.tensor(frame.convert("RGB")).float() / 255.0
        tensor = tensor * 2 - 1
        tensor = tensor.permute(2, 0, 1)
        processed.append(tensor)
    return processed
