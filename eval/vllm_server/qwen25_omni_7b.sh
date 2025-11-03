# multi-GPU (tensor parallel)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Omni-7B \
    --port 8000 --host 127.0.0.1 --dtype bfloat16 -tp 2 \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \


# CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-Omni-7B \
#   --trust-remote-code \
#   --tensor-parallel-size 2 \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.9 \
#   --allowed-local-media-path / \
#   --host 0.0.0.0 --port 8000
