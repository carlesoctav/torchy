from configs.bench.async_grpo_lora_hf_repro.qwen2_5_math_1_5b_r1_b128 import get_config as _get_config  # type: ignore # fallback import

import importlib.util
from pathlib import Path


def get_config():
    config_path = Path(__file__).parent / "async-grpo-lora-hf-repro" / "qwen2.5-math-1.5b-r1-b128.py"
    spec = importlib.util.spec_from_file_location("bench_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()
