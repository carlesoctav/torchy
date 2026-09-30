"""SmolDataEnvs as a TRL GRPO environment: tasks, tables, a program runner, a reward.

A task is a question about real Kaggle tables with one known answer. The
policy writes one Python program; the runner executes it next to the tables;
the last line the program prints is the answer; the dataset's own grader
(``grader.py``, vendored) compares it to the gold. No model grades anything.

Mimics the whileai recipe (recipes/01-simulate/smol-data-envs/env.py) with
one adaptation for TRL: the recipe marks a rollout the environment could not
grade as ``None`` (never ``0.0``), but ``GRPOTrainer`` needs a dense float
reward. So tasks whose tables fail to download are dropped when the dataset
is built, and the reward only ever sees gradeable rollouts. A shell answer
(``echo "2.14"``) still earns zero, whatever it prints.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from unsloth_agent.env.grader import grade

DATASET = "FineEnvs/SmolDataEnvs"
#: The dataset commit this env was written against; the vendored grader in
#: grader.py is the same revision.
REVISION = "b2bf35647e2381b1ab12c2ad7862cbbe4e8f857b"
HF = "https://huggingface.co"

#: Wall-clock seconds one program may run. The dataset's tables are up to a
#: few hundred MB of CSV; pandas reads the largest in well under a minute.
RUN_TIMEOUT_S = 60.0
#: Tries per file. The Hugging Face CDN resets a connection now and then; a
#: table that still fails drops its task from the dataset, it never scores 0.
DOWNLOAD_ATTEMPTS = 3

SYSTEM = (
    "You are a data analyst. You answer a question about data files by writing "
    "one Python program. The program runs once, with the files in ./input, and "
    "pandas and numpy installed. Inspect what you need inside the program. "
    "The last line the program prints is your answer: a bare number (no commas "
    "or units, keep decimal precision), a short label, yes/no, or a "
    "comma-separated list. Reply with the program in a single ```python block."
)

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_SHELL_LANGS = {"bash", "sh", "shell", "zsh", "console", "shell-session"}


def prompt_for(task: dict[str, Any]) -> str:
    """The user turn: the question and the files the program will find."""
    files = "\n".join(f"- input/{name}" for name in task["files"])
    return f"Files:\n{files}\n\nQuestion: {task['question']}"


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


def fetch_tasks(
    split: str,
    cache_dir: str,
    limit: int,
    revision: str,
    dataset: str,
) -> list[dict[str, Any]]:
    """One split of the dataset, with tables on disk. Tasks with no tables to
    read, or whose tables fail to download, are dropped: nothing can grade
    them, and scoring them 0 would punish the policy for our network."""
    import pandas as pd

    raw = Path(cache_dir)
    path = _download(
        f"{HF}/datasets/{dataset}/resolve/{revision}/data/{split}-00000-of-00001.parquet",
        raw / f"{split}.parquet",
    )
    df = pd.read_parquet(path)
    tasks = []
    for rec in df.to_dict("records"):
        rec["files"] = [str(f) for f in rec["files"]]
        if not rec["files"]:
            continue
        tasks.append(rec)
    tasks.sort(key=lambda t: t["task_id"])
    if limit:
        tasks = tasks[:limit]

    kept = []
    for task in tasks:
        dest = raw / "tables" / task["bucket_prefix"]
        try:
            for name in task["files"]:
                url = f"{HF}/buckets/{task['hf_bucket']}/resolve/{task['bucket_prefix']}/{name}"
                _download(url, dest / name)
        except Exception as exc:
            print(f"dropping {task['task_id']}: {type(exc).__name__}: {exc}"[:200], flush=True)
            continue
        task["table_dir"] = str(dest)
        kept.append(task)
    print(f"kept {len(kept)} of {len(tasks)} tasks from {split}", flush=True)
    return kept


def build_dataset(tasks: list[dict[str, Any]]):
    """Tasks into a GRPO dataset: ``prompt`` chat messages plus the grading
    columns TRL forwards to the reward function as kwargs."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt_for(task)},
                ]
                for task in tasks
            ],
            "task_id": [str(task["task_id"]) for task in tasks],
            "reference": [str(task["answer"]) for task in tasks],
            "files": [list(task["files"]) for task in tasks],
            "table_dir": [str(task["table_dir"]) for task in tasks],
            "reward_mode": [str(task.get("reward_mode") or "") for task in tasks],
            "atol": [float(task.get("atol") or 0.0) for task in tasks],
            "rtol": [float(task.get("rtol") or 0.0) for task in tasks],
        }
    )


def split_answer(text: str) -> tuple[str, str]:
    """(language, code) of the last fenced block, else ("", whole text)."""
    blocks = _FENCE.findall(str(text or ""))
    if blocks:
        lang, code = blocks[-1]
        return lang.lower(), code.strip()
    return "", str(text or "").strip()


def run_program(
    code: str, table_dir: str, files: list[str], *, timeout: float = RUN_TIMEOUT_S
):
    """Run ``code`` in a fresh temp directory with the tables copied into
    ./input. Returns (printed answer or None, why).

    A local subprocess with a timeout: it stops runaway loops, and it is not
    a security boundary. Run a policy you do not trust inside a container or
    a sandbox service.
    """
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input"
        inp.mkdir()
        for name in files:
            shutil.copy(Path(table_dir) / name, inp / name)
        prog = Path(tmp) / "solution.py"
        prog.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(prog)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:.0f}s"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return None, f"program failed: {err[-1] if err else f'exit {proc.returncode}'}"[:200]
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None, "program printed nothing"
    return lines[-1], "ran"


def score_completion(
    completion: str,
    reference: str,
    files: list[str],
    table_dir: str,
    reward_mode: str,
    atol: float,
    rtol: float,
    *,
    timeout: float = RUN_TIMEOUT_S,
) -> float:
    """1 when the completion's program prints the gold under the dataset's
    grader, else 0 (wrong, crashed, or answered in shell)."""
    lang, code = split_answer(completion)
    if lang in _SHELL_LANGS or not code:
        return 0.0
    printed, _ = run_program(code, table_dir, files, timeout=timeout)
    if printed is None:
        return 0.0
    return grade(
        str(reference), printed, reward_mode=reward_mode, abs_tol=atol, rel_tol=rtol
    ).reward


def make_reward_fn(timeout: float, num_workers: int):
    """A TRL reward function over one GRPO group. Completions are graded in a
    thread pool; each run is an isolated subprocess in its own temp dir."""

    def grade_one(args: tuple) -> float:
        return score_completion(*args, timeout=timeout)

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rows = zip(
            completions,
            kwargs["reference"],
            kwargs["files"],
            kwargs["table_dir"],
            kwargs["reward_mode"],
            kwargs["atol"],
            kwargs["rtol"],
            strict=True,
        )
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            return list(pool.map(grade_one, rows))

    reward_fn.__name__ = "smoldataenvs"
    return reward_fn


def make(
    split: str = "train",
    cache_dir: str = "data/smoldataenvs",
    limit: int = 0,
    timeout: float = RUN_TIMEOUT_S,
    num_workers: int = 8,
    revision: str = REVISION,
    dataset: str = DATASET,
) -> dict[str, Any]:
    """Fetch tasks (dropping ungradeable ones) and return the GRPO dataset
    plus reward functions. ``limit`` caps to the first N tasks by id; 0 keeps
    the whole split."""
    tasks = fetch_tasks(split, cache_dir, limit, revision, dataset)
    return {
        "train_dataset": build_dataset(tasks),
        "reward_funcs": [make_reward_fn(timeout, num_workers)],
    }
