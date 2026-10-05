from __future__ import annotations

import json
import os
from pathlib import Path

import sws
from transformers import TrainerCallback

from unsloth_agent.env import make_env
from unsloth_agent.sws_utils import run as sws_run
from unsloth_agent.patch import patch_save_tokenizer



def main(config: sws.FinalConfig):
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1" if config.get("unsloth_standby") else "0"
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    patch_save_tokenizer()

    from unsloth_agent.callback import BucketSyncCallback

    env_dict = make_env(config.env_name, config.env.to_dict())

    model, tokenizer = FastLanguageModel.from_pretrained(**config.model.to_dict())
    model = FastLanguageModel.get_peft_model(model, **config.lora.to_dict())

    args = GRPOConfig(**config.grpo.to_dict())


    config_json = Path(f"{config.output_dir}/config.json")
    config_json.parent.mkdir(parents = True, exist_ok=True)

    with config_json.open("w") as f:
        f.write(config.to_json())

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
        trainer.save_model()
    finally:
        bucket_sync.shutdown()


if __name__ == "__main__":
    sws_run(main)
