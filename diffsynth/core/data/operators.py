import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, fps=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.fps = fps  # e.g. 16

    def adjust_to_mod_condition(self, n):
        """Adjust n to largest number <= n such that n % factor == remainder"""
        # General: n' = n - ((n - remainder) % factor)
        # But if n < remainder, return remainder (but we clamp to min=1)
        if n < self.time_division_remainder:
            return max(1, self.time_division_remainder)  # at least 1 frame
        return n - ((n - self.time_division_remainder) % self.time_division_factor)

    def get_resampled_frame_indices(self, total_frames, original_fps, target_fps, max_target_frames=None):
        """
        Return list of frame indices after resampling to target_fps.
        Ensures indices are unique, sorted, and within [0, total_frames-1].
        """
        if total_frames == 0:
            return []

        # Compute video duration in seconds
        duration = total_frames / original_fps

        # Number of frames we *could* sample at target_fps (including t=0)
        # time points: 0, 1/fps, 2/fps, ..., up to <= duration
        max_possible = int(duration * target_fps) + 1  # +1 for t=0

        if max_target_frames is not None:
            max_possible = min(max_possible, max_target_frames)

        frame_times = [i / target_fps for i in range(max_possible)]
        # Map each time to original frame index
        indices = [round(t * original_fps) for t in frame_times]

        # Clamp to valid range
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]

        # Remove duplicates (due to rounding) while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        return unique_indices

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        meta = reader.get_meta_data()
        total_frames = int(reader.count_frames())
        original_fps = meta['fps']

        # Step 1: Determine target FPS
        target_fps = self.fps if self.fps is not None else original_fps

        # Step 2: Get frame indices after resampling
        if self.fps is not None and abs(target_fps - original_fps) > 1e-6:
            # Resample needed
            frame_indices = self.get_resampled_frame_indices(
                total_frames=total_frames,
                original_fps=original_fps,
                target_fps=target_fps,
                max_target_frames=self.num_frames  # try not to exceed desired num_frames
            )
        else:
            # No resampling: use first min(num_frames, total_frames) frames
            frame_indices = list(range(min(self.num_frames, total_frames)))

        # Step 3: Adjust total number of frames to satisfy n ≡ remainder (mod factor)
        actual_num = len(frame_indices)
        adjusted_num = self.adjust_to_mod_condition(actual_num)

        # Safety: if adjusted_num is 0 (e.g., remainder=1 but actual_num=0), fallback to 1 if possible
        if adjusted_num <= 0:
            adjusted_num = 1 if actual_num >= 1 else 0

        # Truncate to adjusted_num frames (take evenly spaced? or just first ones?)
        # Since frame_indices is already temporally ordered, we take first `adjusted_num`
        # But for better temporal coverage, we could subsample uniformly.
        # Here we do uniform subsampling if needed:
        if adjusted_num < len(frame_indices):
            step = len(frame_indices) / adjusted_num
            selected_idx = [int(i * step) for i in range(adjusted_num)]
            frame_indices = [frame_indices[i] for i in selected_idx]
        elif adjusted_num > len(frame_indices):
            # should not happen due to adjust logic, but safety
            adjusted_num = len(frame_indices)
            frame_indices = frame_indices[:adjusted_num]

        # Step 4: Load and process frames
        frames = []
        for frame_id in frame_indices:
            try:
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to read frame {frame_id} from {data}: {e}")
                continue

        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, fps=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.fps = fps
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        images = iio.imread(data, mode="RGB")
        total_frames = len(images)
        
        # Get original fps if fps parameter is set
        original_fps = None
        if self.fps is not None:
            try:
                props = iio.improps(data)
                original_fps = props.get('fps', None)
            except:
                pass
        
        frames = []
        
        # Calculate frame indices based on fps resampling
        if self.fps is not None and original_fps is not None and original_fps != self.fps:
            # Calculate resampling factor
            resample_factor = original_fps / self.fps
            frame_indices = [int(i * resample_factor) for i in range(num_frames) if int(i * resample_factor) < total_frames]
            frame_indices = frame_indices[:num_frames]
        else:
            frame_indices = list(range(min(num_frames, total_frames)))
        
        for frame_id in frame_indices:
            frame = Image.fromarray(images[frame_id])
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio
