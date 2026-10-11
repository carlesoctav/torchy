"""Single-step RL training run with AsyncGRPO + LoRA across HF Storage Bucket.
Mimics: https://huggingface.co/blog/asyncgrpo-lora-hfjobs
Verifies that the race condition for LoRA weight sync is resolved.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time

import torch
from datasets import load_dataset
from math_verify import parse, verify
from peft import LoraConfig
from transformers import AutoTokenizer

from torchy.patch import wait_lora_patch

# Apply our wait_lora_patch to eliminate the weight sync race condition (initial_delay=0.0 to catch 404 and wait)
wait_lora_patch(max_wait=300, retry_interval=2.0, initial_delay=0.0)

import trl.experimental.async_grpo.async_grpo_trainer as async_grpo_trainer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

# Untie lm_head as described in the HF blog
_create_model_from_path = async_grpo_trainer.create_model_from_path


def _create_model_untied(model_id, **kwargs):
    kwargs["attn_implementation"] = "sdpa"
    model = _create_model_from_path(model_id, **kwargs)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False
    return model


async_grpo_trainer.create_model_from_path = _create_model_untied

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Math-1.5B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/content/lora/test-sanity-lora")
SERVE_URL = os.environ.get("SERVE_URL", "http://127.0.0.1:8000")

# Oat math normalization and boxed extraction as in the blog
_SUBS = [
    (r"\left", ""),
    (r"\right", ""),
    (r"\!", ""),
    (r"\,", ""),
    (r"\;", ""),
    (r"\ ", ""),
    (r"\$", ""),
    ("$", ""),
    (r"\%", ""),
    ("%", ""),
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"^\circ", ""),
    (r"\circ", ""),
    (r"\text{", "{"),
    (r"\mbox{", "{"),
]


def _normalize(s: str) -> str:
    s = str(s).strip()
    for old, new in _SUBS:
        s = s.replace(old, new)
    s = re.sub(r"\\frac\s+", r"\\frac", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\s+", r"\\sqrt", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.removesuffix(".0")


def _last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    open_brace = text.find("{", start)
    if open_brace == -1:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i]
    return None


def qwen_math_reward(completions, solution, completion_ids, **kwargs):
    on_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if on_main_thread else None
    verify_timeout = 5 if on_main_thread else None
    if not on_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    rewards = []
    for completion, gold, ids in zip(completions, solution, completion_ids, strict=True):
        answer = _last_boxed(completion[0]["content"])
        if answer is None:
            rewards.append(0.0)
        elif _normalize(answer) == _normalize(gold):
            rewards.append(1.0)
        else:
            parsed_gold = parse(gold, parsing_timeout=parsing_timeout)
            parsed_answer = parse(answer, parsing_timeout=parsing_timeout)
            rewards.append(
                float(
                    bool(parsed_gold)
                    and bool(parsed_answer)
                    and bool(verify(parsed_gold, parsed_answer, timeout_seconds=verify_timeout))
                )
            )
    return rewards


def format_sample(sample):
    return {"prompt": sample["prompt"], "solution": sample["reward_model"]["ground_truth"]}


def main():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("torchy").addHandler(handler)
    logging.getLogger("torchy").setLevel(logging.INFO)
    trl_logger = logging.getLogger("trl")
    trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)

    max_steps = int(os.environ.get("MAX_STEPS", "100"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "8"))
    weight_sync_steps = int(os.environ.get("WEIGHT_SYNC_STEPS", "2"))
    max_completion_length = int(os.environ.get("MAX_COMPLETION_LENGTH", "1024"))
    save_steps = int(os.environ.get("SAVE_STEPS", "20"))
    split = os.environ.get("DATASET_SPLIT", "train")

    print(f"[setup] Loading dataset sail/Sanity-Test-R1D-1.5B split={split} ...", flush=True)
    raw_dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split=split)
    dataset = raw_dataset.map(format_sample, remove_columns=raw_dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.response_template = {
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*|$", "content": "text"}},
    }

    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR,
        bf16=True,
        num_generations=num_generations,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=max_completion_length,
        learning_rate=4e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_staleness=4,
        max_inflight_tasks=16,
        queue_maxsize=64,
        weight_sync_steps=weight_sync_steps,
        max_steps=max_steps,
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps,
        vllm_server_base_url=SERVE_URL,
        report_to="none",
    )

    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=qwen_math_reward,
        peft_config=LoraConfig(r=1, lora_alpha=2, target_modules="all-linear"),
    )

    print("[train] Starting training step...", flush=True)
    trainer.train()
    print(f"[success] Single step RL finished! global_step={trainer.state.global_step}", flush=True)


if __name__ == "__main__":
    main()
