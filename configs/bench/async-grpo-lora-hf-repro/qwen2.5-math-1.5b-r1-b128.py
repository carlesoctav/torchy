import sws


def get_config():
    """Benchmark config for async GRPO LoRA reproduction checkpoints.

    Evaluates Qwen/Qwen2.5-Math-1.5B checkpoints against vLLM server
    on the test split of sail/Sanity-Test-R1D-1.5B (AIME 2024 & 2025).
    """
    config = sws.Config()

    config.exp_name = "qwen2.5-math-1.5b-r1-b128"
    config.project_name = "async-grpo-lora-hf-repro"
    config.run_name = lambda: f"{config.project_name}_{config.exp_name}"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.checkpoints_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.output_dir = lambda: f"outputs/bench/{config.project_name}/{config.exp_name}"
    config.seed = 3407

    # Model configuration
    config.model.model_name = "Qwen/Qwen2.5-Math-1.5B"
    config.model.vllm_server_base_url = "http://localhost:8000"

    # Evaluation parameters
    config.eval.max_completion_length = 3000
    config.eval.temperature = 0.6
    config.eval.top_p = 0.95
    config.eval.concurrency = 8
    config.eval.eval_base_model = True
    config.eval.log_completions = True
    config.eval.num_completions_to_print = 5
    config.eval.chat_template_kwargs = {"enable_thinking": True}
    config.eval.report_to = "trackio"

    # Direct top-level shortcuts for flexibility
    config.model_name = lambda: config.model.model_name
    config.vllm_server_base_url = lambda: config.model.vllm_server_base_url
    config.max_completion_length = lambda: config.eval.max_completion_length
    config.concurrency = lambda: config.eval.concurrency
    config.log_completions = lambda: config.eval.log_completions
    config.chat_template_kwargs = lambda: config.eval.chat_template_kwargs
    config.report_to = lambda: config.eval.report_to

    # Environment configuration: sail/Sanity-Test-R1D-1.5B with test split
    config.env_name = "sanity_math"
    config.env.dataset_name = "sail/Sanity-Test-R1D-1.5B"
    config.env.split = "test"
    config.env.model_id = lambda: config.model.model_name
    config.env.max_completion_length = lambda: config.eval.max_completion_length
    config.env.limit = 0

    return config
