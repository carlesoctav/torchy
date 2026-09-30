import sws


def get_config():
    config = sws.Config()

    config.exp_name = "qwen3-4b-grpo-smoldata"
    config.project_name = "unsloth-agent"
    config.output_dir = lambda: f"outputs/{config.project_name}/{config.exp_name}"
    config.seed = 3407

    # Base weights load in bf16 (load_in_4bit=False); only the LoRA adapter trains.
    config.model.model_name = "unsloth/Qwen3-4B"
    config.model.max_seq_length = 4096
    config.model.dtype = "bfloat16"
    config.model.load_in_4bit = False
    # fast_inference shares the weights with the vLLM rollout engine instead
    # of holding a second copy; standby hands vLLM's KV-cache memory back to
    # training while the optimizer step runs. gpu_memory_utilization can then
    # stay at its max: Unsloth handles the rest.
    config.model.fast_inference = True
    config.model.max_lora_rank = 32
    config.model.gpu_memory_utilization = 0.9
    config.model.standby = True

    config.lora.r = 32
    config.lora.lora_alpha = 64
    config.lora.lora_dropout = 0.0
    config.lora.target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config.lora.bias = "none"
    config.lora.use_gradient_checkpointing = "unsloth"
    config.lora.random_state = lambda: config.seed

    config.grpo.output_dir = lambda: config.output_dir
    config.grpo.run_name = lambda: config.exp_name
    config.grpo.report_to = "trackio"
    config.grpo.seed = lambda: config.seed
    config.grpo.bf16 = lambda: config.model.dtype == "bfloat16"
    config.grpo.fp16 = lambda: config.model.dtype == "float16"
    config.grpo.num_generations = 8
    config.grpo.per_device_train_batch_size = lambda: config.grpo.num_generations
    config.grpo.gradient_accumulation_steps = 1
    config.grpo.max_prompt_length = 2048
    config.grpo.max_completion_length = 2048
    config.grpo.max_steps = 500
    config.grpo.learning_rate = 5e-6
    config.grpo.weight_decay = 0.0
    config.grpo.warmup_steps = 0
    config.grpo.lr_scheduler_type = "constant"
    config.grpo.optim = "adamw_8bit"
    config.grpo.max_grad_norm = 1.0
    config.grpo.temperature = 0.9
    config.grpo.top_p = 1.0
    config.grpo.beta = 0.0
    config.grpo.epsilon = 0.2
    config.grpo.use_vllm = False
    config.grpo.log_completions = True
    config.grpo.num_completions_to_print = 4
    config.grpo.logging_steps = 1
    config.grpo.save_steps = 100
    config.grpo.save_total_limit = 2

    config.env_name = "smoldataenvs"
    config.env.split = "train"
    config.env.cache_dir = "data/smoldataenvs"
    config.env.limit = 0
    config.env.timeout = 60.0
    config.env.num_workers = 8

    config.trackio.project = lambda: config.project_name
    config.trackio.space_id = None

    return config
