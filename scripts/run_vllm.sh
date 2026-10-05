mycolab run -s vllm --gpu l4 \
    --dir .:/content/unsloth-agent \
    -v carlesoctav/multi-serverasync-grpo:/content/lora \
    -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
    -e VLLM_SERVER_DEV_MODE=1 \
    -- uv run vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 --logprobs-mode processed_logprobs --generation-config vllm \
        --enable-lora --max-lora-rank 1 --max-loras 6
