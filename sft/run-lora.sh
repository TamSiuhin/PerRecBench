CUDA_VISIBLE_DEVICES=0,2 \
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./deepspeed_zero3.yaml ./run_sft.py ./config_lora.yaml
