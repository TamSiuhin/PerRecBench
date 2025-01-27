CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
--api_key EMPTY \
--enable-lora \
--lora-modules perrecbench-lora-mix=/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-llama-3.1-8b-sft-lora-mix perrecbench-lora-avg=/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-llama-3.1-8b-sft-lora-avg perrecbench-lora-point=/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-llama-3.1-8b-sft-lora-point perrecbench-lora-group=/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-llama-3.1-8b-sft-lora-group perrecbench-lora-pair=/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/sft/model/PerRecBench-llama-3.1-8b-sft-lora-pair \
--served-model-name PerRecBench-llama-3.1-8b-sft-lora \
--tensor-parallel-size 1 \
--port 8002 \
--max-model-len 40000