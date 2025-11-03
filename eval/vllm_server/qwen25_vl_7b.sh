# multi-GPU (tensor parallel)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --allowed-local-media-path / \
  --host 0.0.0.0 --port 8000
