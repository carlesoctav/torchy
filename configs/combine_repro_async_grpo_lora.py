from __future__ import annotations

import sws


def get_config():
    """Combined configuration for reproducing AsyncGRPO + LoRA across 2 Colab VMs.

    Defines:
    1. config.infra.inference -> ColabSession config for inference VM.
    2. config.infra.train -> ColabSession config for trainer VM.
    3. config.inference -> vLLM server config.
    4. config.model, config.grpo, config.env -> Trainer & Environment settings.
    """
    config = sws.Config()

    # ==========================================
    # 1. Experiment & Project Meta
    # ==========================================
    config.exp_name = "qwen2.5-math-1.5b-r1"
    config.project_name = "async-grpo-lora-hf-repro"
    config.bucket_id = "carlesoctav/multi-serverasync-grpo"
    config.output_dir = lambda: f"/content/lora/{config.project_name}/{config.exp_name}"
    config.seed = 3407

    # ==========================================
    # 2. Infrastructure: ColabSession for Inference
    # ==========================================
    config.infra.inference.session = "vllm"
    config.infra.inference.gpu = "l4"
    config.infra.inference.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.infra.inference.local_forwards = ["8000:localhost:8000"]
    config.infra.inference.stop_vm = False

    # ==========================================
    # 3. Infrastructure: ColabSession for Trainer
    # ==========================================
    config.infra.train.session = "trainer"
    config.infra.train.gpu = "l4"
    config.infra.train.volumes = ["carlesoctav/multi-serverasync-grpo:/content/lora"]
    config.infra.train.remote_forwards = ["8000:localhost:8000"]
    config.infra.train.stop_vm = False

    # Common remote project directory
    config.infra.torchy_dir = "/content/torchy"

    # ==========================================
    # 4. Inference: vLLM Server Config
    # ==========================================
    config.inference.model = "Qwen/Qwen2.5-Math-1.5B"
    config.inference.host = "0.0.0.0"
    config.inference.port = 8000
    config.inference.max_model_len = 4096
    config.inference.gpu_memory_utilization = 0.85
    config.inference.max_lora_rank = 1
    config.inference.max_loras = 6
    config.inference.logprobs_mode = "processed_logprobs"
    config.inference.generation_config = "vllm"
    config.inference.enable_lora = True
    config.inference.env_vars = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        "VLLM_SERVER_DEV_MODE": "1",
    }

    # ==========================================
    # 5. Model & LoRA Config
    # ==========================================
    config.model.model = "Qwen/Qwen2.5-Math-1.5B"
    config.model.peft_config.r = 1
    config.model.peft_config.lora_alpha = 2
    config.model.peft_config.target_modules = "all-linear"

    # ==========================================
    # 6. GRPO Trainer Config
    # ==========================================
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

    # ==========================================
    # 7. Environment & Dataset Config
    # ==========================================
    config.env_name = "sanity_math"
    config.env.dataset_name = "sail/Sanity-Test-R1D-1.5B"
    config.env.split = "train"
    config.env.model_id = lambda: config.model.model
    config.env.max_completion_length = lambda: config.grpo.max_completion_length
    config.env.limit = 0

    return config
