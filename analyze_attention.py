import os
os.environ.setdefault("MLLM_NO_CFG", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator

from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_inter_3 import (
    WanVideoUnit_BlockScheduler,
    WanVideoUnit_MLLMEmbedder_MetaQuery,
    sample_frames_with_constraints,
)
from diffsynth.pipelines.wan_video_autoregressive_query import (
    WanVideoAutoregressiveQueryPipeline,
    ModelConfig,
)
from diffsynth.core.data.unified_dataset import WanVideoInterDataset
from diffsynth.core.loader.file import load_state_dict


NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch/multi-GPU: infer video first, then analyze query attention.")
    parser.add_argument("--jsonl_path", type=str, default="/root/workspace/zzt/data/AgiBotWorld-Alpha/agibot_result_sample.jsonl")
    parser.add_argument("--base_path", type=str, default="")
    parser.add_argument("--sample_idx", type=int, default=-1, help="Only run this dataset index. -1 means batch mode.")
    parser.add_argument("--only_indices", type=str, default="", help="Comma-separated dataset indices, e.g. 1,5,9")
    parser.add_argument("--max_items", type=int, default=0, help="Max assigned items per rank. 0 means no limit.")
    parser.add_argument("--block_idx", type=int, default=-1, help="Which block to analyze. -1 means all blocks.")

    parser.add_argument("--target_fps", type=int, default=6)
    parser.add_argument("--source_fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max_frames", type=int, default=241)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--mllm_cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--tiled", action="store_true")

    parser.add_argument("--use_gt_mllm", action="store_true", help="Use dataset input_video as MLLM history during inference.")
    parser.add_argument("--use_gt_vae", action="store_true", help="Use dataset input_video as VAE clean latents during inference.")
    parser.add_argument("--gt_decode", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_metaqueries", type=int, default=64)
    parser.add_argument("--mllm_model_path", type=str, default="/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--wan_dit_path", type=str, default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--wan_t5_path", type=str, default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--wan_vae_path", type=str, default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--wan_tokenizer_path", type=str, default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")

    parser.add_argument("--pre_lora_path", type=str, default="", help="Optional base LoRA path.")
    parser.add_argument("--lora_path", type=str, default="", help="Optional target LoRA path.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--output_dir", type=str, default="query_attention_outputs")
    parser.add_argument("--topk", type=int, default=40)

    parser.add_argument("--save_inferred_video", action="store_true")
    parser.add_argument("--save_fps", type=int, default=15)
    parser.add_argument("--save_quality", type=int, default=5)
    return parser.parse_args()


def to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def parse_only_indices(s: str) -> List[int]:
    if not s.strip():
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def load_lora_weights(pipe, lora_path: str, device: torch.device):
    if not lora_path:
        return False
    if not os.path.exists(lora_path):
        print(f"[LoRA] not found, skip: {lora_path}")
        return False

    state_dict = load_state_dict(lora_path, torch_dtype=pipe.torch_dtype, device=device)
    dit_state_dict = {k.replace("dit.", ""): v for k, v in state_dict.items() if k.startswith("dit.")}
    mllm_state_dict = {k.replace("mllm_encoder.", ""): v for k, v in state_dict.items() if k.startswith("mllm_encoder.")}

    if not dit_state_dict:
        dit_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mllm_encoder.")}

    pipe.dit.load_state_dict(dit_state_dict, strict=False)
    pipe.mllm_encoder.load_state_dict(mllm_state_dict, strict=False)
    pipe.load_lora(pipe.dit, state_dict=dit_state_dict, alpha=1.0)
    print(f"[LoRA] loaded: {lora_path}")
    return True


def build_block_info(pipe, prompt_list: List[str], clip_frames: List[int], num_frames: int):
    scheduler = WanVideoUnit_BlockScheduler()
    return scheduler.process(pipe, prompt_list, clip_frames, num_frames)["block_info"]


def collect_history_videos(video_frames, block_info, current_block: int):
    all_videos = []
    all_metadata = []

    for block_idx in range(1, current_block + 1):
        block = block_info[block_idx - 1]
        block_frames = video_frames[block["start_frame"]:block["end_frame"]]
        local_indices = sample_frames_with_constraints(len(block_frames), target_stride=8)
        global_indices = [block["start_frame"] + i for i in local_indices]
        sampled_frames = [video_frames[i] for i in global_indices] if global_indices else []

        all_videos.append(sampled_frames)
        all_metadata.append(
            {
                "fps": 16,
                "frames_indices": global_indices,
                "total_num_frames": len(video_frames),
            }
        )

    return all_videos, all_metadata


def build_full_text(prompt_list: List[str], block_info, current_block: int, num_metaqueries: int) -> str:
    system_prompt = WanVideoUnit_MLLMEmbedder_MetaQuery().system_prompt
    query_tokens = ["<|begin_of_query|>"] + [f"<|query_{i}|>" for i in range(num_metaqueries)] + ["<|end_of_query|>"]
    query_str = "".join(query_tokens)

    full_text_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    prev_prompt_idx = None

    for block_idx in range(current_block + 1):
        block = block_info[block_idx]
        prompt_idx = block["prompt_idx"]
        user_content_parts = []

        if prompt_idx != prev_prompt_idx:
            user_content_parts.append(prompt_list[prompt_idx])
            prev_prompt_idx = prompt_idx

        if block_idx > 0:
            if user_content_parts:
                user_content_parts.append(" ")
            # Keep exact spacing behavior with pipeline.encode_mllm_for_block_metaquery
            user_content_parts.append(" <|vision_start|><|video_pad|><|vision_end|>")

        user_content = "".join(user_content_parts)
        full_text_parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        full_text_parts.append(f"<|im_start|>assistant\n{query_str}<|im_end|>\n")

    return "".join(full_text_parts)


def prepare_last_layer_qk(pipe, model_inputs, position_ids):
    lm = pipe.mllm_encoder.model.language_model
    last_layer = lm.layers[-1]
    captured = {}

    def _capture_last_layer_input(module, args):
        captured["hidden_before_last"] = args[0]

    hook = last_layer.register_forward_pre_hook(_capture_last_layer_input)
    try:
        with torch.inference_mode():
            _ = pipe.mllm_encoder.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=position_ids,
                pixel_values=model_inputs.get("pixel_values", None),
                pixel_values_videos=model_inputs.get("pixel_values_videos", None),
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                rope_deltas=model_inputs.get("rope_deltas", None),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        hook.remove()

    if "hidden_before_last" not in captured:
        raise RuntimeError("Failed to capture hidden states before last MLLM layer.")

    with torch.inference_mode():
        hidden_before_last = captured["hidden_before_last"]
        cache_position = torch.arange(hidden_before_last.shape[1], device=hidden_before_last.device)
        text_position_ids = position_ids[0]
        causal_mask = create_causal_mask(
            config=lm.config,
            input_embeds=hidden_before_last,
            attention_mask=model_inputs["attention_mask"],
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )

        cos, sin = lm.rotary_emb(hidden_before_last, position_ids)
        h = last_layer.input_layernorm(hidden_before_last)
        sa = last_layer.self_attn
        input_shape = h.shape[:-1]
        hidden_shape = (*input_shape, -1, sa.head_dim)

        q = sa.q_norm(sa.q_proj(h).view(hidden_shape)).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(h).view(hidden_shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, sa.num_key_value_groups)

    return {
        "q": q,
        "k": k,
        "causal_mask": causal_mask,
        "scaling": sa.scaling,
    }


def compute_query_scores_from_qk(prepared, query_positions: List[int]):
    q = prepared["q"]          # (1, H, S, D)
    k = prepared["k"]          # (1, H, S, D)
    causal_mask = prepared["causal_mask"]  # (1, 1, S, S) additive mask
    scaling = prepared["scaling"]

    with torch.inference_mode():
        device = q.device
        qpos = torch.tensor(query_positions, dtype=torch.long, device=device)

        q_sel = q[0, :, qpos, :]  # (H, Qq, D)
        k_all = k[0]              # (H, S, D)
        attn_logits = torch.matmul(q_sel, k_all.transpose(1, 2)) * scaling  # (H, Qq, S)

        if causal_mask is not None:
            mask_rows = causal_mask[0, 0, qpos, :k_all.shape[1]]  # (Qq, S)
            attn_logits = attn_logits + mask_rows.unsqueeze(0)

        attn_weights = torch.softmax(attn_logits.float(), dim=-1)  # (H, Qq, S)
        key_idx = torch.arange(attn_weights.shape[-1], device=device)[None, :]  # (1, S)
        valid = key_idx < qpos[:, None]  # (Qq, S), only previous tokens
        valid_hq = valid[None, :, :]     # (1, Qq, S)
        num_heads = attn_weights.shape[0]
        # numerator sums over (H, Q), so denominator must also include H.
        denom = (valid_hq.sum(dim=(0, 1)) * num_heads).clamp_min(1)
        scores = (attn_weights * valid_hq).sum(dim=(0, 1)) / denom
        scores = scores.clamp_(0.0, 1.0)

    return scores.detach().cpu()


def get_assigned_indices(total_items: int, rank: int, world_size: int, args) -> List[int]:
    if args.sample_idx >= 0:
        if args.sample_idx >= total_items:
            return []
        return [args.sample_idx] if (args.sample_idx % world_size == rank) else []

    only_indices = parse_only_indices(args.only_indices)
    if only_indices:
        filtered = [i for i in only_indices if 0 <= i < total_items]
        assigned = [i for i in filtered if i % world_size == rank]
    else:
        assigned = list(range(rank, total_items, world_size))

    if args.max_items > 0:
        assigned = assigned[:args.max_items]
    return assigned


def analyze_one_item(pipe, args, device, output_dir: Path, idx: int, item: dict, rank: int):
    prompt_list = item["prompt_list"]
    clip_frames = item["clip_frames"]
    gt_input_video = item.get("video", None)
    num_frames = sum(clip_frames)

    negative_prompt_list = [NEG_PROMPT] * len(prompt_list)

    print(f"[Rank {rank}] infer idx={idx} ...")
    inferred_video = pipe(
        prompt_list=prompt_list,
        negative_prompt_list=negative_prompt_list,
        clip_frames=clip_frames,
        input_video=gt_input_video if (args.use_gt_mllm or args.use_gt_vae) else None,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        use_mllm_condition=True,
        use_gt_mllm=args.use_gt_mllm,
        use_gt_vae=args.use_gt_vae,
        gt_decode=args.gt_decode,
        cfg_scale=args.cfg_scale,
        mllm_cfg_scale=args.mllm_cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        tiled=args.tiled,
    )

    block_info = build_block_info(pipe, prompt_list, clip_frames, num_frames)
    if args.block_idx >= 0:
        if args.block_idx >= len(block_info):
            raise ValueError(f"Invalid block_idx={args.block_idx}, total_blocks={len(block_info)}")
        analyze_blocks = [args.block_idx]
    else:
        analyze_blocks = list(range(len(block_info)))

    tokenizer = pipe.mllm_processor.tokenizer
    video_token_id = pipe.mllm_encoder.config.video_token_id
    image_token_id = pipe.mllm_encoder.config.image_token_id
    video_id = item.get("video_id", f"idx_{idx}")
    safe_video_id = str(video_id).replace("/", "_")

    # Build a single full-sequence MLLM input (up to last block), then reuse it for all block queries.
    full_block = len(block_info) - 1
    all_videos, all_metadata = collect_history_videos(inferred_video, block_info, full_block)
    full_text = build_full_text(prompt_list, block_info, full_block, pipe.mllm_encoder.num_metaqueries)

    if all_videos:
        model_inputs = pipe.mllm_processor(
            text=[full_text],
            videos=all_videos,
            padding=True,
            video_metadata=all_metadata,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        ).to(device)
    else:
        model_inputs = pipe.mllm_processor(
            text=[full_text],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)

    position_ids, _ = pipe.mllm_encoder.model.get_rope_index(
        input_ids=model_inputs["input_ids"],
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
    )

    input_ids = model_inputs["input_ids"][0]
    token_ids = input_ids.tolist()
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    pos = position_ids[:, 0, :].detach().cpu()

    boq = pipe.mllm_encoder.boq_token_id
    eoq = pipe.mllm_encoder.eoq_token_id
    boq_positions = (input_ids == boq).nonzero(as_tuple=True)[0].tolist()
    eoq_positions = (input_ids == eoq).nonzero(as_tuple=True)[0].tolist()
    if not boq_positions or not eoq_positions:
        raise RuntimeError("No MetaQuery segment found in input_ids.")

    prepared_qk = prepare_last_layer_qk(pipe, model_inputs, position_ids)

    for current_block in analyze_blocks:
        target_seg = min(current_block, len(boq_positions) - 1, len(eoq_positions) - 1)
        b_pos = boq_positions[target_seg]
        e_pos = eoq_positions[target_seg]
        if e_pos <= b_pos + 1:
            raise RuntimeError("MetaQuery segment has no query tokens.")

        query_positions = list(range(b_pos + 1, e_pos))
        scores = compute_query_scores_from_qk(prepared_qk, query_positions)

        out_rows = []
        for i in range(scores.shape[0]):
            token_id = int(token_ids[i])
            token_text = token_texts[i]
            is_vision = token_id in (video_token_id, image_token_id)
            thw = None
            if is_vision:
                thw = [int(pos[0, i].item()), int(pos[1, i].item()), int(pos[2, i].item())]
            out_rows.append(
                {
                    "index": i,
                    "token_id": token_id,
                    "token_text": token_text,
                    "score": float(scores[i].item()),
                    "score_percent": float(scores[i].item() * 100.0),
                    "is_vision_token": bool(is_vision),
                    "thw": thw,
                }
            )

        out_rows = [r for r in out_rows if r["index"] <= e_pos]

        stem = f"rank{rank}_idx{idx}_{safe_video_id}_block{current_block}"
        json_path = output_dir / f"{stem}_query_attention.json"
        tsv_path = output_dir / f"{stem}_query_attention.tsv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_idx": idx,
                    "video_id": video_id,
                    "current_block": current_block,
                    "query_span": [b_pos + 1, e_pos - 1],
                    "num_query_tokens": len(query_positions),
                    "num_tokens_reported": len(out_rows),
                    "inference": {
                        "seed": args.seed,
                        "num_inference_steps": args.num_inference_steps,
                        "cfg_scale": args.cfg_scale,
                        "mllm_cfg_scale": args.mllm_cfg_scale,
                        "sigma_shift": args.sigma_shift,
                        "use_gt_mllm": args.use_gt_mllm,
                        "use_gt_vae": args.use_gt_vae,
                    },
                    "rows": out_rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("index\ttoken_id\ttoken_text\tscore_percent\tscore\tis_vision_token\tthw\n")
            for r in out_rows:
                thw_text = "" if r["thw"] is None else f"({r['thw'][0]},{r['thw'][1]},{r['thw'][2]})"
                f.write(
                    f"{r['index']}\t{r['token_id']}\t{r['token_text']}\t{r['score_percent']:.4f}%\t{r['score']:.8f}\t{int(r['is_vision_token'])}\t{thw_text}\n"
                )

        topk = min(args.topk, len(out_rows))
        top_rows = sorted(out_rows, key=lambda x: x["score"], reverse=True)[:topk]
        print(f"[Rank {rank}] done idx={idx} block={current_block}: {json_path.name}")
        for r in top_rows[:3]:
            thw_text = "" if r["thw"] is None else f" thw={tuple(r['thw'])}"
            print(f"  idx={r['index']:4d} score={r['score_percent']:.4f}% tok={r['token_text']}{thw_text}")

    if args.save_inferred_video:
        video_stem = f"rank{rank}_idx{idx}_{safe_video_id}"
        video_path = output_dir / f"{video_stem}_inferred.mp4"
        save_video(inferred_video, str(video_path), fps=args.save_fps, quality=args.save_quality)


def main():
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    torch_dtype = to_dtype(args.torch_dtype)

    os.environ["MLLM_QEURY"] = str(args.num_metaqueries)

    dataset = WanVideoInterDataset(
        base_path=args.base_path,
        metadata_path=args.jsonl_path,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        height=args.height,
        width=args.width,
        num_frames=args.max_frames,
    )

    total_items = len(dataset.data) if hasattr(dataset, "data") else len(dataset)
    indices = get_assigned_indices(total_items, rank, world_size, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Rank {rank}] Assigned {len(indices)} / {total_items} items.")

    pipe = WanVideoAutoregressiveQueryPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=args.wan_dit_path),
            ModelConfig(path=args.wan_t5_path),
            ModelConfig(path=args.wan_vae_path),
            ModelConfig(path=[
                os.path.join(args.mllm_model_path, "model-00001-of-00002.safetensors"),
                os.path.join(args.mllm_model_path, "model-00002-of-00002.safetensors"),
            ]),
        ],
        tokenizer_config=ModelConfig(path=args.wan_tokenizer_path),
        mllm_processor_config=ModelConfig(path=args.mllm_model_path),
        num_metaqueries=args.num_metaqueries,
    )

    if args.pre_lora_path:
        load_lora_weights(pipe, args.pre_lora_path, device)
    if args.lora_path:
        load_lora_weights(pipe, args.lora_path, device)

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception as e:
            print(f"[Rank {rank}] failed load idx={idx}: {e}")
            continue

        try:
            analyze_one_item(pipe, args, device, output_dir, idx, item, rank)
        except Exception as e:
            print(f"[Rank {rank}] failed idx={idx}: {e}")
            continue

    accelerator.wait_for_everyone()
    if rank == 0:
        print("All done.")


if __name__ == "__main__":
    main()
