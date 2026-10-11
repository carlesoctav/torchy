import sws


def get_config():
    config = sws.Config()

    config.exp_name = "qwen3-1.7b"
    config.project_name = "wordle_bench"
    config.output_dir = lambda: f"outputs/{config.project_name}/{config.exp_name}"
    config.parquet_filename = "wordle_benchmark.parquet"
    config.seed = 3407

    config.model.model_name = "Qwen/Qwen3-1.7B"
    config.model.vllm_server_base_url = "http://localhost:8000"

    config.eval.max_completion_length = 2048
    config.eval.temperature = 0.6
    config.eval.top_p = 0.95
    config.eval.max_tool_calling_iterations = 10
    config.eval.concurrency = 8
    config.eval.log_completions = True
    config.eval.num_completions_to_print = 10
    config.eval.chat_template_kwargs = {"enable_thinking": True}

    # Direct top-level shortcuts for flexibility
    config.model_name = lambda: config.model.model_name
    config.vllm_server_base_url = lambda: config.model.vllm_server_base_url
    config.max_completion_length = lambda: config.eval.max_completion_length
    config.concurrency = lambda: config.eval.concurrency
    config.log_completions = lambda: config.eval.log_completions
    config.chat_template_kwargs = lambda: config.eval.chat_template_kwargs

    config.env_name = "openenv_wordle"
    config.env.dataset_size = 100
    config.env.env_url = "http://localhost:8001"

    return config
