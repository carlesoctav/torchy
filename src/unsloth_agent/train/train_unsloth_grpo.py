from __future__ import annotations

import json
import os
from pathlib import Path

import sws
from transformers import TrainerCallback

from unsloth_agent.env import make_env
from unsloth_agent.sws_utils import run as sws_run
from unsloth_agent.patch.unpatch import unpatch_all



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
    unpatch_all()

    from unsloth_agent.callback import BucketSyncCallback

    env_dict = make_env(config.env_name, config.env.to_dict())

    model, tokenizer = FastLanguageModel.from_pretrained(**config.model.to_dict())
    model = FastLanguageModel.get_peft_model(model, **config.lora.to_dict())

    args = GRPOConfig(**config.grpo.to_dict())


    config_json = Path(f"{config.output_dir}/config.json")
    config_json.parent.mkdir(parents = True, exist_ok=True)

    with config_json.open("w") as f:
        f.write(config.to_json())

    # bucket_sync = BucketSyncCallback(config)
    # callbacks: list[TrainerCallback] = [bucket_sync]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        # callbacks=callbacks,
        **env_dict
    )
    try:
        trainer.train()
    finally:
        if config.get("repo_id"):
            model.push_to_hub_merged(f"{config.repo_id}", tokenizer, save_method = "merged_16bit")
        else:
            model.save_pretrained_merged(f"{config.output_dir}/merged", tokenizer, save_method = "merged_16bit")
        bucket_sync.shutdown()


if __name__ == "__main__":
    sws_run(main)
