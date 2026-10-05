from __future__ import annotations

from pathlib import Path

import sws

from unsloth_agent.env import make_env
from unsloth_agent.sws_utils import run as sws_run


def main(config: sws.FinalConfig):
    from unsloth_agent.patch import wait_lora_patch

    wait_lora_patch()

    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


    env_dict = make_env(config.env_name, config.env.to_dict())
    model_dict = config.model.to_dict()
    args = AsyncGRPOConfig(**config.grpo.to_dict())

    config_json = Path(f"{config.output_dir}/config.json")
    config_json.parent.mkdir(parents = True, exist_ok=True)
    with config_json.open("w") as f:
        f.write(config.to_json())


    trainer = AsyncGRPOTrainer(
        **model_dict,
        args=args,
        # callbacks=callbacks,
        **env_dict
    )
    trainer.train()


if __name__ == "__main__":
    sws_run(main)

