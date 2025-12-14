import torch
from torch import nn
from typing import List, Optional, Sequence, Union

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ..utils.data import MLLM_SYSTEM_PROMPT, video_to_pil_frames


class WanMLLMEncoder(nn.Module):
    def __init__(self, model_path: str, torch_dtype: torch.dtype = torch.bfloat16, enable_gradient_checkpointing: bool = False, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.device = device
        self.processor = Qwen3VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype, device_map=None)
        if device is not None:
            self.model.to(device)
        if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return self

    def build_prompt(self, text_prompt: str) -> str:
        return MLLM_SYSTEM_PROMPT + text_prompt + "<|im_end|>"

    @torch.no_grad()
    def encode(self, text_prompt: Union[str, List[str]], video_frames: Union[Sequence, torch.Tensor]) -> Optional[torch.Tensor]:
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = text_prompt
        videos = [video_to_pil_frames(video_frames)]
        chat_prompts = [self.build_prompt(p) for p in prompts]
        inputs = self.processor(text=chat_prompts, videos=videos, return_tensors="pt")
        if self.device is not None:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return self.extract_multimodal_embeddings(hidden_states, inputs.get("attention_mask"))

    def extract_multimodal_embeddings(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            max_len = attention_mask.sum(dim=1).max().item()
            hidden_states = hidden_states[:, :max_len]
        return hidden_states
