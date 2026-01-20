from .operators import *
import torch, json, pandas, random


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        cfg_drop=0.0,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.cfg_drop = cfg_drop
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        fps=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    fps=fps,
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    fps=fps,
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
            if random.random() < self.cfg_drop:
                if "context" in data[1]:
                    data[1]["context"] = torch.zeros_like(data[1]["context"])
                if "prompt_embeddings_map" in data[0]:
                    for k, v in data[0]["prompt_embeddings_map"].items():
                        data[0]["prompt_embeddings_map"][k] = torch.zeros_like(v)
        else:
            data = self.data[data_id % len(self.data)].copy()
            if "video" in data:
                data["video_id"] = data["video"].split("/")[-1].removesuffix(".mp4")
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class WanVideoInterDataset(UnifiedDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        cfg_drop=0.0,
        target_fps=16,
        source_fps=30,
        max_pixels=1920*1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
        num_frames=81,
    ):
        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=data_file_keys,
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
            cfg_drop=cfg_drop,
        )
        self.target_fps = target_fps
        self.source_fps = source_fps
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = ImageCropAndResize(
            height, width, max_pixels, height_division_factor, width_division_factor
        )
        self.max_num_frames = num_frames

    def _resolve_path(self, path):
        if self.base_path and not os.path.isabs(path):
            return os.path.join(self.base_path, path)
        return path

    def _convert_frame_index(self, frame_idx):
        return int(round(frame_idx * self.target_fps / self.source_fps))

    def _round_clip_length(self, length, is_first):
        if is_first:
            k = int(round((length - 1) / 4))
            if k < 0:
                k = 0
            return max(1, 4 * k + 1)
        k = int(round(length / 4))
        if k < 1:
            k = 1
        return max(4, 4 * k)

    def _build_prompt_and_clips(self, data):
        prompt_list = data["detailed_action_captions"]
        action_config = data["input"]["label_info"]["action_config"]
        count = min(len(prompt_list), len(action_config))
        prompt_list = prompt_list[:count]
        clip_frames = []
        for idx in range(count):
            cfg = action_config[idx]
            start_frame = cfg["start_frame"]
            end_frame = cfg["end_frame"]
            if idx == 0:
                start_frame = 0
            start_t = self._convert_frame_index(start_frame)
            end_t = self._convert_frame_index(end_frame)
            if end_t <= start_t:
                end_t = start_t + 1
            clip_len = self._round_clip_length(end_t - start_t, is_first=(idx == 0))
            clip_frames.append(clip_len)
        return prompt_list, clip_frames

    def _adjust_clip_frames(self, clip_frames, total_frames):
        diff = total_frames - sum(clip_frames)
        if diff == 0 or not clip_frames:
            return clip_frames
        clip_frames[-1] += diff
        min_len = 1 if len(clip_frames) == 1 else 4
        if clip_frames[-1] < min_len:
            clip_frames[-1] = min_len
        return clip_frames

    def _truncate_clips_to_max_frames(self, prompt_list, clip_frames, max_frames):
        if max_frames is None or max_frames <= 0 or not clip_frames:
            return prompt_list, clip_frames
        if sum(clip_frames) <= max_frames:
            return prompt_list, clip_frames

        kept_prompts = []
        kept_frames = []
        total = 0
        for prompt, frames in zip(prompt_list, clip_frames):
            if total + frames > max_frames:
                break
            kept_prompts.append(prompt)
            kept_frames.append(frames)
            total += frames

        if kept_frames:
            return kept_prompts, kept_frames

        # Fallback: keep the first clip but clamp its length to fit the cap.
        limit = max(1, max_frames)
        k = (limit - 1) // 4
        first_len = max(1, 4 * k + 1)
        if first_len > limit:
            first_len = limit
        return [prompt_list[0]], [first_len]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            return super().__getitem__(data_id)

        data = self.data[data_id % len(self.data)].copy()
        if (
            "input" in data
            and isinstance(data["input"], dict)
            and "path" in data["input"]
            and "detailed_action_captions" in data
        ):
            prompt_list, clip_frames = self._build_prompt_and_clips(data)
            max_frames = self.max_num_frames
            prompt_list, clip_frames = self._truncate_clips_to_max_frames(
                prompt_list, clip_frames, max_frames
            )
            video_path = self._resolve_path(data["input"]["path"])
            desired_num_frames = sum(clip_frames)
            video_loader = LoadVideo(
                num_frames=desired_num_frames,
                time_division_factor=self.time_division_factor,
                time_division_remainder=self.time_division_remainder,
                frame_processor=self.frame_processor,
                fps=self.target_fps,
            )
            video = video_loader(video_path)
            clip_frames = self._adjust_clip_frames(clip_frames, len(video))
            data["video"] = video
            data["video_id"] = "-".join(data["input"]["path"].split("/")[-4:-2])
            data["prompt_list"] = prompt_list
            data["clip_frames"] = clip_frames
            return data

        return super().__getitem__(data_id)
