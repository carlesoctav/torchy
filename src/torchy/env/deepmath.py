"""DeepMath-103K as a TRL GRPO environment: math prompts, a symbolic reward.

Each row is a competition-style math problem with a known solution. The
policy answers in free text; the reward parses the gold and the completion's
``\\boxed{}`` answer with math-verify and checks symbolic equivalence. No
model grades anything.

Mimics the TRL GRPO quickstart (``trl-lib/DeepMath-103K`` +
``trl.rewards.accuracy_reward``) with one adaptation shared with the
smoldataenvs env: TRL's reward marks a rollout whose gold does not parse as
``None``, but this repo keeps rewards dense, so rows whose gold fails to
parse are dropped when the dataset is built. The filtered rows are cached
per revision, so the 97k-row parse only happens once. A wrong or
unparseable answer still earns zero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from trl.rewards import accuracy_reward

DATASET = "trl-lib/DeepMath-103K"
#: The dataset commit this env was written against.
REVISION = "066c50a88d4e14cefc056e31111db2dba17f6c68"
HF = "https://huggingface.co"

#: Tries per file. The Hugging Face CDN resets a connection now and then.
DOWNLOAD_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step, then put your "
    "put your answer on \\boxed{}"
)


def _math_verify():
    """math-verify's parse/verify names, or a TRL-style install hint."""
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as exc:
        raise ImportError(
            "Please install the `math-verify` package to use the deepmath env: "
            "`pip install math-verify`"
        ) from exc
    return NormalizationConfig, LatexExtractionConfig, parse, verify


def _download(url: str, dest: Path) -> Path:
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            break
        except requests.ConnectionError:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    part.replace(dest)
    return dest


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    """Parquet-shaped prompt column into chat messages.

    The dataset stores ``prompt`` as a list of ``{role, content}`` dicts;
    pandas hands it back as a numpy array of dicts. Plain strings and
    missing values degrade to a single user turn and to no turns.
    """
    if prompt is None:
        return []
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    try:
        messages = list(prompt)
    except TypeError:
        return []
    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "") or "")
        else:
            role, content = "user", str(msg or "")
        if not content.strip():
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _gold_parses(solution: str) -> bool:
    _, _, parse, _ = _math_verify()
    try:
        return len(parse(solution, parsing_timeout=10)) != 0
    except Exception:
        return False


def fetch_tasks(
    split: str,
    cache_dir: str,
    limit: int,
    revision: str,
    dataset: str,
) -> list[dict[str, Any]]:
    """One split of the dataset, minus rows nothing can grade.

    Rows with an empty prompt or solution, or whose gold solution math-verify
    cannot parse, are dropped: TRL would score them ``None`` (skipped), and
    this repo would rather not spend rollouts on them. The kept rows are
    cached per split, limit, and revision; a cache hit skips the download
    and the parse filter alike.
    """
    import pandas as pd

    raw = Path(cache_dir)
    filtered = raw / f"filtered-{split}-limit{limit}-{revision[:8]}.parquet"
    if filtered.exists() and filtered.stat().st_size > 0:
        df = pd.read_parquet(filtered)
        return [
            {"prompt": json.loads(rec["prompt"]), "solution": str(rec["solution"])}
            for rec in df.to_dict("records")
        ]

    path = _download(
        f"{HF}/datasets/{dataset}/resolve/{revision}/data/{split}-00000-of-00001.parquet",
        raw / f"{split}.parquet",
    )
    df = pd.read_parquet(path)
    rows = []
    for rec in df.to_dict("records"):
        prompt = normalize_prompt(rec.get("prompt"))
        solution = str(rec.get("solution") or "").strip()
        if not prompt or not solution:
            continue
        rows.append({"prompt": prompt, "solution": solution})
    if limit:
        rows = rows[:limit]

    kept = [row for row in rows if _gold_parses(row["solution"])]
    print(
        f"kept {len(kept)} of {len(rows)} tasks from {split} "
        f"({len(rows) - len(kept)} with unparseable gold dropped)",
        flush=True,
    )
    pd.DataFrame(
        [
            {"prompt": json.dumps(row["prompt"]), "solution": row["solution"]}
            for row in kept
        ]
    ).to_parquet(filtered)
    return kept


def build_dataset(tasks: list[dict[str, Any]]):
    """Tasks into a GRPO dataset: ``prompt`` chat messages plus the ``solution``
    column TRL forwards to the reward function as a kwarg."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [task["prompt"] for task in tasks],
            "solution": [task["solution"] for task in tasks],
        }
    )


def completion_text(completion: Any) -> str:
    """The text of one completion. TRL passes completions as one-message
    chat lists for conversational datasets; plain strings pass through, so
    direct calls work too."""
    if isinstance(completion, str):
        return completion
    try:
        first = list(completion)[0]
    except (TypeError, IndexError):
        return str(completion or "")
    if isinstance(first, dict):
        return str(first.get("content", "") or "")
    return str(first or "")


def _score_with_detail(completion: Any, solution: str) -> tuple[float | None, str, str]:
    """(reward, gold_parsed, answer_parsed), mirroring
    ``trl.rewards.accuracy_reward``: symbolic equivalence of the gold and the
    completion's ``\\boxed{}`` answer, ``None`` when the gold does not parse
    (unreachable after :func:`fetch_tasks` filters, kept for TRL parity)."""
    import logging
    import threading

    NormalizationConfig, LatexExtractionConfig, parse, verify = _math_verify()

    content = completion_text(completion)

    # math_verify uses signal.alarm() for timeouts, which only works in the
    # main thread. Disable timeouts elsewhere to avoid ValueError.
    is_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = None if not is_main_thread else 10
    verify_timeout = None if not is_main_thread else 5
    if not is_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    gold_parsed = parse(solution, parsing_timeout=parsing_timeout)
    if len(gold_parsed) == 0:
        return None, "[unparseable]", "[skipped]"
    # The answer must be provided in correct latex (no malformed operators),
    # with \\boxed{} tried first.
    answer_parsed = parse(
        content,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(units=True),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
        parsing_timeout=parsing_timeout,
    )
    reward = float(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))
    return (
        reward,
        str(gold_parsed),
        str(answer_parsed) if answer_parsed else "[unparseable]",
    )


def score_completion(completion: Any, solution: str) -> float | None:
    """1 when the completion's boxed answer matches the gold symbolically,
    else 0 (wrong or unparseable answer), else None (unparseable gold)."""
    reward, _, _ = _score_with_detail(completion, solution)
    return reward


def make_reward_fn():
    """A TRL reward function over one GRPO group, with the trainer's
    ``log_extra`` columns (gold/answer parses) when the trainer provides it."""

    def reward_fn(prompts, completions, solution, log_extra=None, **kwargs):
        rewards, gold_strs, answer_strs = [], [], []
        for completion, sol in zip(completions, solution, strict=True):
            reward, gold_str, answer_str = _score_with_detail(completion, sol)
            rewards.append(reward)
            gold_strs.append(gold_str)
            answer_strs.append(answer_str)
        if log_extra is not None:
            log_extra("solution", list(solution))
            log_extra("gold_parsed", gold_strs)
            log_extra("answer_parsed", answer_strs)
        return rewards

    reward_fn.__name__ = "deepmath"
    return reward_fn


def make(
    split: str = "train",
    cache_dir: str = "data/deepmath",
    limit: int = 0,
    revision: str = REVISION,
    dataset: str = DATASET,
) -> dict[str, Any]:
    """Fetch tasks (dropping ungradeable rows) and return the GRPO dataset
    plus reward functions. ``limit`` caps to the first N rows; 0 keeps the
    whole split. Needs ``math-verify`` installed (``pip install math-verify``)."""
    dataset = load_dataset("trl-lib/DeepMath-103K", split=split)

    def add_system_prompt(batch):
        batch["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}] + batch["prompt"]
        return batch

    dataset = dataset.map(add_system_prompt, num_proc = 8)


    return {
        "train_dataset": dataset,
        "reward_funcs": [accuracy_reward],
    }
