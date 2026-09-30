from __future__ import annotations

import os

import sws

from unsloth_agent.callback import BucketSyncCallback
from unsloth_agent.env import make_env
from unsloth_agent.sws_utils import run as sws_run


def main(config: sws.FinalConfig):
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1" if config.get("unsloth_standby") else "0"
    # Sybal
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    env = make_env(config.env_name, config.env.to_dict())

    model, tokenizer = FastLanguageModel.from_pretrained(**config.model.to_dict())
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
            "'<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
    model = FastLanguageModel.get_peft_model(model, **config.lora.to_dict())

    args = GRPOConfig(**config.grpo.to_dict())

    callbacks = [BucketSyncCallback(config)]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=env["reward_funcs"],
        args=args,
        train_dataset=env["train_dataset"],
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    sws_run(main)
