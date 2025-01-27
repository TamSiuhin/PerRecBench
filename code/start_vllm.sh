CUDA_VISIBLE_DEVICES=0 \
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
--api_key EMPTY \
--served-model-name Mistral-7B-Instruct-v0.3 \
--tensor-parallel-size 1 \
--port 8002 \
# --max-model-len 20000