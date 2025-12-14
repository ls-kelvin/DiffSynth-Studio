from .unified_dataset import UnifiedDataset
from ..utils.data import sample_mllm_frames


class VideoDataset(UnifiedDataset):
    def __init__(self, *args, include_mllm_frames: bool = False, mllm_fps: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_mllm_frames = include_mllm_frames
        self.mllm_fps = mllm_fps

    def __getitem__(self, data_id):
        data = super().__getitem__(data_id)
        if self.include_mllm_frames and isinstance(data, dict) and "video" in data:
            data["mllm_frames"] = sample_mllm_frames(data["video"], num_frames=len(data["video"]), mllm_fps=self.mllm_fps)
        return data
