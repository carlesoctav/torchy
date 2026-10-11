#!/usr/bin/env bash
# Spawn a vLLM server on Colab runtime using mycolab run with the shared HF Storage Bucket mounted.
# Mimics: https://huggingface.co/blog/asyncgrpo-lora-hfjobs
set -euo pipefail

SESSION=${SESSION:-vllm}
GPU=${GPU:-l4}
BUCKET=${BUCKET:-carlesoctav/multi-serverasync-grpo}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_LORA_RANK=${MAX_LORA_RANK:-1}
MAX_LORAS=${MAX_LORAS:-6}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

echo "=== Spawning vLLM on Colab session '${SESSION}' with GPU '${GPU}' ==="
echo "Model: ${MODEL_ID}"
echo "Bucket: ${BUCKET} mounted at /content/lora"
echo "Port: ${PORT}"

mycolab run -s "${SESSION}" --gpu "${GPU}" \
    --dir .:/content/torchy \
    -v "${BUCKET}:/content/lora" \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -- uv run vllm serve "${MODEL_ID}" \
        --host 0.0.0.0 --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --logprobs-mode processed_logprobs \
        --generation-config vllm \
        --enable-lora --max-lora-rank "${MAX_LORA_RANK}" --max-loras "${MAX_LORAS}"
