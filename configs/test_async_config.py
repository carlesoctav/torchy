import sws
from peft import LoraConfig


def get_config():
    """
    please becareful we dont use unsloth here
    """

    config = sws.Config()

    config.exp_name = "async-l4-l4-qwen3-1.7b-grpo-wordle"
    config.project_name = "wordle-qwen3"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407


    config.model.model = "Qwen/Qwen3-1.7B" 
    config.model.peft_config = LoraConfig(r = 1, lora_alpha=2, target_modules = "all-linear")

    config.grpo.run_name = lambda : f"{config.project_name}_{config.exp_name}"
    config.grpo.project = lambda : config.project_name
    config.grpo.output_dir = lambda: config.output_dir

    config.grpo.per_device_train_batch_size = 8
    config.grpo.gradient_accumulation_steps = 4
    # config.grpo.beta = 0.001
    # config.grpo.learning_rate = 5e-5

    # config.grpo.vllm_gpu_memory_utilization = 0.3
    # config.grpo.vllm_max_model_length = 4096
    # config.grpo.use_vllm = True

    # Whole multi-turn episode must fit: game messages + all guesses.
    config.grpo.max_completion_length = 1024
    config.grpo.max_steps = 3000
    config.grpo.save_steps = 100
    config.grpo.weight_decay = 0.0
    config.grpo.warmup_steps = 0.05


    config.grpo.vllm_server_base_url = "http://localhost:8000" 
    config.grpo.max_staleness = 4
    config.grpo.weight_sync_steps = 4

    config.grpo.report_to = "trackio"
    config.grpo.log_completions = True
    config.grpo.num_completions_to_print = 2
    config.grpo.chat_template_kwargs = {"enable_thinking": False}

    config.env_name = "openenv_wordle"
    config.env.dataset_size = 1000
    config.env.env_url = "http://localhost:8001"

    return config
