from __future__ import annotations

import os
from pathlib import Path

import sws
from transformers import TrainerCallback

from unsloth_agent.env import make_env
from unsloth_agent.sws_utils import run as sws_run


def main(config: sws.FinalConfig):
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1" if config.get("unsloth_standby") else "0"
    # Before any heavy import: trackio reads TRACKIO_DIR once at import time.
    # Project level (sibling of the exp dir) so one dashboard shows all runs.
    os.environ["TRACKIO_DIR"] = str(Path(config.output_dir).resolve().parent / "trackio")
    # Unsloth first: it patches TRL's vLLM integration at import, so importing
    # trl beforehand binds the pristine trainer classes and fails on vllm's
    # moved symbols.
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    # FP8 on GPUs without native vLLM block-FP8 kernels (e.g. L4 / sm89): vLLM 0.30
    # picks Marlin, which repacks the weights into int32 and breaks Unsloth's shared
    # FP8 weights. Force the Triton block-FP8 kernel instead. Equivalent to
    # `--linear-backend triton`, which Unsloth does not forward to vLLM.
    # Set `config.vllm_linear_backend = "triton"` in the config to enable.
    # NOTE: also clear ~/.cache/vllm/torch_compile_cache if it was built with Marlin.
    vllm_linear_backend = config.get("vllm_linear_backend")
    if vllm_linear_backend:
        import vllm.model_executor.kernels.linear as _vllm_linear

        _vllm_linear._get_linear_backend = lambda **kw: vllm_linear_backend

    # unsloth_zoo 2026.9.8 passes dtype= to transformers 5.x FP8Linear, which
    # dropped that argument. Drop it here until unsloth_zoo is updated.
    if config.model.to_dict().get("load_in_fp8"):
        import transformers.integrations.finegrained_fp8 as _fp8

        _orig_fp8_init = _fp8.FP8Linear.__init__
        _fp8.FP8Linear.__init__ = lambda self, *a, dtype=None, **k: _orig_fp8_init(self, *a, **k)

    from unsloth_agent.callback import BucketSyncCallback

    env_dict = make_env(config.env_name, config.env.to_dict())

    model, tokenizer = FastLanguageModel.from_pretrained(**config.model.to_dict())
    # if not getattr(tokenizer, "chat_template", None):
    #     # TRL matches templates by exact string to parse tool calls; fall
    #     # back to its known Qwen3 template, not a hand-rolled one.
    from trl.chat_template_utils import qwen3_chat_template

    tokenizer.chat_template = qwen3_chat_template
    model = FastLanguageModel.get_peft_model(model, **config.lora.to_dict())

    args = GRPOConfig(**config.grpo.to_dict())

    bucket_sync = BucketSyncCallback(config)
    callbacks: list[TrainerCallback] = [bucket_sync]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        callbacks=callbacks,
        **env_dict
    )
    try:
        trainer.train()
    finally:
        bucket_sync.shutdown()
    trainer.save_model()


if __name__ == "__main__":
    sws_run(main)

