"""The sws config loads, finalizes, and honors CLI overrides."""

import pytest
import sws

from unsloth_agent.sws_utils import load_config_builder


def test_grpo_config_finalizes():
    builder = load_config_builder("configs/grpo_smoldataenvs.py")
    assert isinstance(builder, sws.Config)
    config, unused = builder.finalize([], return_unused_argv=True)
    assert unused == []
    assert config.exp_name == "qwen3-4b-grpo-smoldata"
    assert config.output_dir == "outputs/unsloth-agent/qwen3-4b-grpo-smoldata"
    assert config.grpo.per_device_train_batch_size == config.grpo.num_generations


def test_grpo_config_overrides_propagate_through_lambdas():
    builder = load_config_builder("configs/grpo_smoldataenvs.py")
    config, _ = builder.finalize(
        ["grpo.num_generations=16", "model.model_name=unsloth/Qwen3-8B"],
        return_unused_argv=True,
    )
    assert config.grpo.per_device_train_batch_size == 16
    assert config.model.model_name == "unsloth/Qwen3-8B"


def test_train_module_imports_without_gpu_deps():
    import unsloth_agent.train.train_grpo as train_grpo

    assert callable(train_grpo.main)


@pytest.mark.parametrize(
    "config_path, expected_print",
    [
        ("configs/grpo_smoldataenvs.py", 4),
        ("configs/grpo_deepmath.py", 0),
        ("configs/qwen_vllm_grpo.py", 0),
    ],
)
def test_grpo_configs_enable_completion_logging(config_path, expected_print):
    builder = load_config_builder(config_path)
    config, unused = builder.finalize([], return_unused_argv=True)
    assert unused == []
    assert config.grpo.log_completions is True
    assert config.grpo.num_completions_to_print == expected_print
