import sws

try:
    from peft import LoraConfig
except ImportError:
    class LoraConfig:  # type: ignore
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


def get_config():
    """Config reproducing https://huggingface.co/blog/asyncgrpo-lora-hfjobs

    Uses Qwen/Qwen2.5-Math-1.5B, sail/Sanity-Test-R1D-1.5B, LoRA r=1 alpha=2,
    syncing adapters across the shared HF Storage Bucket mounted at /content/lora.
    Reports directly to Trackio at https://trackio.intelligences.systems.
    """
    config = sws.Config()

    config.exp_name = "qwen2.5-math-1.5b-r1"
    config.project_name = "async-grpo-lora-hf-repro"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407

    # Model configuration
    config.model.model = "Qwen/Qwen2.5-Math-1.5B"
    config.model.peft_config = LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules="all-linear",
    )

    # GRPO / Async GRPO parameters (matching HF blog reference)
    config.grpo.run_name = lambda: f"{config.project_name}_{config.exp_name}"
    config.grpo.project = lambda: config.project_name
    config.grpo.output_dir = lambda: config.output_dir
    config.grpo.bf16 = True
    config.grpo.learning_rate = 4e-5
    config.grpo.lr_scheduler_type = "constant"
    config.grpo.adam_beta2 = 0.95
    config.grpo.num_generations = 8
    config.grpo.temperature = 1.0
    config.grpo.top_p = 1.0
    config.grpo.max_completion_length = 3000
    config.grpo.per_device_train_batch_size = 32
    config.grpo.gradient_accumulation_steps = 4
    config.grpo.gradient_checkpointing = True
    config.grpo.max_staleness = 4
    config.grpo.max_inflight_tasks = 128
    config.grpo.queue_maxsize = 512
    config.grpo.weight_sync_steps = 4
    config.grpo.max_steps = 500
    config.grpo.save_strategy = "steps"
    config.grpo.save_steps = 50
    config.grpo.vllm_server_base_url = "http://127.0.0.1:8000"
    config.grpo.report_to = "trackio"
    config.grpo.logging_steps = 1
    config.grpo.log_completions = True
    config.grpo.num_completions_to_print = 2

    # vLLM server configuration (matching HF blog reference: --max-model-len 4096)
    config.vllm.max_model_len = 4096
    config.vllm.gpu_memory_utilization = 0.85
    config.vllm.max_lora_rank = 1
    config.vllm.max_loras = 6

    # Environment configuration
    config.env_name = "sanity_math"
    config.env.dataset_name = "sail/Sanity-Test-R1D-1.5B"
    config.env.split = "train"
    config.env.model_id = lambda: config.model.model
    config.env.max_completion_length = lambda: config.grpo.max_completion_length
    config.env.limit = 0

    return config
