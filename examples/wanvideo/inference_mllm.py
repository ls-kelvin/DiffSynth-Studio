import torch
from diffsynth import WanVideoPipeline, ModelConfig


if __name__ == "__main__":
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=[
            ModelConfig(model_id="Wan/Wan-T2V:transformer/..."),
            ModelConfig(model_id="Wan/Wan-T2V:text_encoder/..."),
            ModelConfig(model_id="Wan/Wan-T2V:vae/..."),
            ModelConfig(path="./models/train/Wan_MLLM_lora"),
            ModelConfig(model_id="Qwen/Qwen3-VL"),
        ],
        torch_dtype=torch.bfloat16,
    )

    video = pipe(
        prompt="A cat playing with a ball",
        use_mllm_condition=True,
        num_frames=65,
        height=720,
        width=1280,
    )

    video.save("output.mp4")
