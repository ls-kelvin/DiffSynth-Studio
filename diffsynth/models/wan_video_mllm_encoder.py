import torch
from torch import nn
from typing import Optional, List

from transformers import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor


class WanMLLMEncoder(nn.Module):
    """
    Qwen3-VL backbone wrapped for DiffSynth. Follows the integration style of Qwen-Image text encoder:
    the model is instantiated from an explicit config instead of using from_pretrained.
    """

    def __init__(self, torch_dtype: torch.dtype = torch.bfloat16, num_metaqueries: int = int(os.getenv("MLLM_QEURY", "0"))):
        super().__init__()
        self.num_metaqueries = num_metaqueries
        config = Qwen3VLConfig(**{
            "architectures": [
                "Qwen3VLForConditionalGeneration"
            ],
            "image_token_id": 151655,
            "model_type": "qwen3_vl",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "dtype": "bfloat16",
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [
                    24,
                    20,
                    20
                ],
                "rope_type": "default"
                },
                "rope_theta": 5000000,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 151936
            },
            "tie_word_embeddings": True,
            "transformers_version": "4.57.0.dev0",
            "video_token_id": 151656,
            "vision_config": {
                "deepstack_visual_indexes": [
                5,
                11,
                17
                ],
                "depth": 24,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1024,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "model_type": "qwen3_vl",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652
            }
        )
        self.model = Qwen3VLModel(config)
        self.config = config
        self.torch_dtype = torch_dtype
        
        # MetaQuery 支持
        self.boq_token_id = None
        self.eoq_token_id = None
        self.query_token_ids = None
        self._original_vocab_size = None
        
        # MetaQuery tokens are initialized after weights are loaded.

    def init_metaquery_tokens_from_tokenizer(self, tokenizer):
        """使用 tokenizer 的 special tokens 初始化 MetaQuery tokens"""
        if self.num_metaqueries <= 0:
            return
        if tokenizer is None:
            raise ValueError("Tokenizer is required to initialize MetaQuery tokens.")

        token_list = ["<|begin_of_query|>", "<|end_of_query|>"]
        token_list.extend([f"<|query_{i}|>" for i in range(self.num_metaqueries)])
        tokenizer.add_special_tokens({"additional_special_tokens": token_list})

        new_vocab_size = len(tokenizer)
        self.model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=128)

        def _token_id(tok: str) -> int:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tokenizer.convert_ids_to_tokens(tok_id) != tok:
                raise ValueError(f"MetaQuery token {tok} is not in tokenizer vocab.")
            return tok_id

        self.boq_token_id = _token_id("<|begin_of_query|>")
        self.eoq_token_id = _token_id("<|end_of_query|>")
        self.query_token_ids = [_token_id(f"<|query_{i}|>") for i in range(self.num_metaqueries)]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_query_token_ids(self) -> torch.LongTensor:
        """返回 query token 序列: [boq_id, query0_id, query1_id, ..., eoq_id]"""
        if self.num_metaqueries <= 0:
            raise ValueError("MetaQuery not enabled")
        if self.boq_token_id is None or self.eoq_token_id is None or self.query_token_ids is None:
            raise ValueError("MetaQuery tokens are not initialized. Call init_metaquery_tokens_from_tokenizer first.")
        ids = [self.boq_token_id] + self.query_token_ids + [self.eoq_token_id]
        return torch.tensor(ids, dtype=torch.long)

    def extract_query_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """从 hidden_states 中提取 query token 对应的部分
        
        Args:
            hidden_states: (B, L, D) MLLM 输出
            input_ids: (B, L) 输入 token ids
        
        Returns:
            query_embeds: (B, num_metaqueries, D)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # 找到 <begin_of_query> 和 <end_of_query> 位置
        query_embeds_list = []
        for b in range(batch_size):
            boq_mask = (input_ids[b] == self.boq_token_id)
            eoq_mask = (input_ids[b] == self.eoq_token_id)
            boq_pos = boq_mask.nonzero(as_tuple=True)[0][0].item()
            eoq_pos = eoq_mask.nonzero(as_tuple=True)[0][0].item()
            
            # 提取 query tokens (在 boq 和 eoq 之间，不包含 boq/eoq)
            query_hidden = hidden_states[b, boq_pos + 1:eoq_pos, :]
            query_embeds_list.append(query_hidden)
        
        query_embeds = torch.stack(query_embeds_list, dim=0)
        return query_embeds

    def freeze_all_except_queries(self):
        """冻结所有参数，只保留 query embedding 可训练"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.num_metaqueries > 0:
            if self.boq_token_id is None or self.eoq_token_id is None or self.query_token_ids is None:
                raise ValueError("MetaQuery tokens are not initialized. Call init_metaquery_tokens_from_tokenizer first.")
            self.model.language_model.embed_tokens.weight.requires_grad = True
            # 注册 hook 确保梯度只更新新增的 token
            def freeze_hook(grad):
                grad = grad.clone()
                trainable_ids = [self.boq_token_id] + self.query_token_ids + [self.eoq_token_id]
                mask = torch.zeros(grad.shape[0], device=grad.device, dtype=torch.bool)
                mask[torch.tensor(trainable_ids, device=grad.device, dtype=torch.long)] = True
                grad[~mask] = 0
                return grad
            self.model.language_model.embed_tokens.weight.register_hook(freeze_hook)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """返回需要训练的参数（只有 query token embeddings）"""
        if self.num_metaqueries > 0:
            return [self.model.language_model.embed_tokens.weight]
        return []

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        return outputs.hidden_states
