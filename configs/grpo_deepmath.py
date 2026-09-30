import sws


def get_config():
    config = sws.Config()

    config.exp_name = "qwen3-4b-grpo-deepmath-new"
    config.project_name = "unsloth-agent"
    config.output_dir = lambda: f"outputs/{config.project_name}/{config.exp_name}"
    config.seed = 3407
    config.unsloth_standby = True

    config.model.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    config.model.max_seq_length = 512+1668
    config.model.dtype = "bfloat16"
    config.model.load_in_4bit = False
    config.model.fast_inference = True
    config.model.gpu_memory_utilization = 0.9

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

    # config.grpo.num_generations = 8
    config.grpo.per_device_train_batch_size = 8
    config.grpo.gradient_accumulation_steps = 4

    config.grpo.max_completion_length = 512
    config.grpo.max_steps = 500
    config.grpo.learning_rate = 5e-6
    config.grpo.weight_decay = 0.0
    config.grpo.warmup_steps = 0
    config.grpo.lr_scheduler_type = "constant"
    config.grpo.optim = "adamw_8bit"
    config.grpo.max_grad_norm = 1.0

    # config.grpo.temperature = 0.9
    # config.grpo.top_p = 1.0
    config.grpo.log_completions = True
    config.grpo.logging_steps = 1
    config.grpo.report_to = "trackio"
    config.grpo.run_name = lambda : f"{config.project_name}_{config.exp_name}"
    config.grpo.project = lambda : config.project_name


    config.env_name = "deepmath"
    config.env.split = "train"
    config.env.cache_dir = "data/deepmath"
    config.env.limit = 0


    return config
