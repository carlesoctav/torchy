import sws


def get_config():
    config = sws.Config()

    config.exp_name = "qwen3-1.7b-grpo-wordle"
    config.project_name = "wordle-qwen3"
    config.bucket_id = lambda: f"carlesoctav/{config.project_name}"
    config.output_dir = lambda: f"outputs/{config.project_name}/{config.exp_name}"
    config.repo_id = "carlesoctav/qwen3-1.7b-wordle"
    config.seed = 3407
    config.unsloth_standby = True

    config.model.model_name = "Qwen/Qwen3-1.7B"
    config.model.use_exact_model_name = True
    config.model.max_seq_length = 2048
    config.model.dtype = "bfloat16"
    config.model.load_in_fp8 = False 
    config.model.load_in_4bit = False
    config.model.fast_inference = True
    config.model.gpu_memory_utilization = 0.9
    config.model.max_lora_rank = 8

    config.lora.r = 8
    config.lora.lora_alpha = 16
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
    config.grpo.gradient_accumulation_steps = 1

    # Whole multi-turn episode must fit: game messages + all guesses.
    config.grpo.max_completion_length = 1024
    config.grpo.max_steps = 3000
    config.grpo.save_steps = 100
    config.grpo.learning_rate = 1e-5
    config.grpo.weight_decay = 0.0
    config.grpo.warmup_steps = 0.05
    # config.grpo.lr_scheduler_type = "linear"
    config.grpo.optim = "adamw_8bit"
    config.grpo.max_grad_norm = 1.0
    config.grpo.beta = 0.001
    # config.grpo.vllm_gpu_memory_utilization = 0.3
    # config.grpo.vllm_max_model_length = 4096
    # config.grpo.use_vllm = True

    config.grpo.run_name = lambda : f"{config.project_name}_{config.exp_name}"
    config.grpo.project = lambda : config.project_name
    config.grpo.output_dir = lambda: config.output_dir

    config.grpo.report_to = "trackio"
    config.grpo.log_completions = True
    config.grpo.num_completions_to_print = 2
    config.grpo.logging_steps = 1
    config.grpo.chat_template_kwargs = {"enable_thinking": False}

    config.env_name = "openenv_wordle"
    config.env.dataset_size = 1000
    config.env.env_url = "http://localhost:8001"

    return config
