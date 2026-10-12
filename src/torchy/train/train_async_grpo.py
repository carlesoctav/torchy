from __future__ import annotations

import logging
import os
from pathlib import Path

import sws

from torchy.env import make_env
from torchy.sws_utils import run as sws_run

logger = logging.getLogger(__name__)


def main(config: sws.FinalConfig):
    # Configure logging so TRL and torchy logs are visible
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    trl_logger = logging.getLogger("trl")
    if not trl_logger.handlers:
        trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)
    ua_logger = logging.getLogger("torchy")
    if not ua_logger.handlers:
        ua_logger.addHandler(handler)
    ua_logger.setLevel(logging.INFO)

    from torchy.patch import wait_lora_patch

    # Apply wait_lora_patch directly on VLLMClient (no proxy needed)
    wait_lora_patch(max_wait=300, retry_interval=2.0, initial_delay=0.0)

    import torch
    import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from transformers.trainer_utils import get_last_checkpoint

    # Untie lm_head from embed_tokens to ensure safe weights and FSDP/SDPA compatibility
    _create_model_from_path = async_grpo_trainer.create_model_from_path

    def _create_model_untied(model_id, **kwargs):
        model = _create_model_from_path(model_id, **kwargs)
        if hasattr(model, "lm_head") and hasattr(model, "config"):
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            model.config.tie_word_embeddings = False
        return model

    async_grpo_trainer.create_model_from_path = _create_model_untied

    env_dict = make_env(config.env_name, config.env.to_dict())
    model_dict = config.model.to_dict()
    args = AsyncGRPOConfig(**config.grpo.to_dict())

    config_json = Path(f"{config.output_dir}/config.json")
    config_json.parent.mkdir(parents=True, exist_ok=True)
    with config_json.open("w") as f:
        f.write(config.to_json())

    trainer = AsyncGRPOTrainer(
        **model_dict,
        args=args,
        **env_dict,
    )

    last_checkpoint = get_last_checkpoint(config.output_dir) if os.path.isdir(config.output_dir) else None
    if last_checkpoint:
        logger.info(f"[resume] Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info(f"[done] Finished training. global_step={trainer.state.global_step}")

    final_dir = os.path.join(config.output_dir, "final-adapter")
    try:
        async_grpo_trainer.save_lora_adapter(
            trainer.accelerator.unwrap_model(trainer.model),
            trainer.accelerator,
            trainer._adapter_name,
            final_dir,
        )
        logger.info(f"[final] Saved final adapter to {final_dir}")
    except Exception as e:
        logger.warning(f"Could not save final adapter to {final_dir}: {e}")


def main_cli():
    sws_run(main)


if __name__ == "__main__":
    main_cli()
