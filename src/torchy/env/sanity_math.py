"""Sanity Math Environment for Async GRPO reproduction of https://huggingface.co/blog/asyncgrpo-lora-hfjobs

Uses `sail/Sanity-Test-R1D-1.5B` and oat's rule-based math verification (`qwen_math_reward`).
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Response template for Qwen2.5-Math / ChatML
QWEN_MATH_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {"content": {"close_pattern": r"<\|im_end\|>\s*|$", "content": "text"}},
}

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


def make_qwen_math_reward(max_completion_length: int = 1024):
    def qwen_math_reward(completions, solution, completion_ids, **kwargs):
        on_main_thread = threading.current_thread() is threading.main_thread()
        parsing_timeout = 10 if on_main_thread else None
        verify_timeout = 5 if on_main_thread else None
        if not on_main_thread:
            logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
            logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

        try:
            from math_verify import parse, verify
        except ImportError:
            parse = None
            verify = None

        rewards = []
        for completion, gold, ids in zip(completions, solution, completion_ids, strict=True):
            if max_completion_length > 0 and len(ids) >= max_completion_length:
                rewards.append(0.0)
                continue
            answer = _last_boxed(completion[0]["content"])
            if answer is None:
                rewards.append(0.0)
            elif _normalize(answer) == _normalize(gold):
                rewards.append(1.0)
            elif parse is not None and verify is not None:
                parsed_gold = parse(gold, parsing_timeout=parsing_timeout)
                parsed_answer = parse(answer, parsing_timeout=parsing_timeout)
                rewards.append(
                    float(
                        bool(parsed_gold)
                        and bool(parsed_answer)
                        and bool(verify(parsed_gold, parsed_answer, timeout_seconds=verify_timeout))
                    )
                )
            else:
                rewards.append(0.0)
        return rewards

    qwen_math_reward.__name__ = "qwen_math_reward"
    return qwen_math_reward


def format_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {"prompt": sample["prompt"], "solution": sample["reward_model"]["ground_truth"]}


def make(
    dataset_name: str = "sail/Sanity-Test-R1D-1.5B",
    split: str = "train",
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    max_completion_length: int = 1024,
    limit: int = 0,
    **kwargs,
) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info(f"Loading dataset {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.response_template = QWEN_MATH_RESPONSE_TEMPLATE

    reward_fn = make_qwen_math_reward(max_completion_length=max_completion_length)

    return {
        "train_dataset": dataset,
        "reward_funcs": reward_fn,
        "processing_class": tokenizer,
    }
