# # hot
# export MLLM_TRANSORMER=0
# # inter_4
# accelerate launch \
#     --num_processes=8 \
#     ar_batch_inference_inter.py \
#     --run_cate mllm_hot --lora_step 5600

# query
export MLLM_QEURY=256
export MLLM_TRANSFORMER_LAYER=12

accelerate launch \
    --num_processes=8 \
    ar_batch_inference_inter.py \
    --run_cate mllm_query --lora_step 7200

# # cfg 
# accelerate launch \
#     --num_processes=8 \
#     ar_batch_inference_inter.py \
#     --run_cate mllm_cfg --lora_step 13600 --mllm_cfg_scale 5.0 --cfg_scale 2.0