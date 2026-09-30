"""Offline tests for the DeepMath env: no network, no GPU.

Scoring tests need math-verify and skip without it; prompt/dataset/cache
tests run anywhere.
"""

import inspect
import json
import sys

import pytest

from unsloth_agent.env import deepmath as env

PROMPT = [{"role": "user", "content": "What is 1/2 + 1/6?"}]
GOLD = r"\frac{2}{3}"
RIGHT = r"Common denominator 6: 3/6 + 1/6. \boxed{\frac{2}{3}}"
WRONG = r"Common denominator 6: 3/6 + 1/6. \boxed{\frac{1}{2}}"
NO_BOX = "The answer is two thirds, I think."
# Prose gold from the dataset's own row 0: nothing symbolic to parse.
PROSE_GOLD = "Yes"


def completion(text):
    return [{"role": "assistant", "content": text}]


def test_normalize_prompt_message_list():
    assert env.normalize_prompt(PROMPT) == PROMPT


def test_normalize_prompt_plain_string():
    assert env.normalize_prompt("Solve x + 1 = 2.") == [
        {"role": "user", "content": "Solve x + 1 = 2."}
    ]


def test_normalize_prompt_missing_is_empty():
    assert env.normalize_prompt(None) == []
    assert env.normalize_prompt(float("nan")) == []


def test_normalize_prompt_skips_empty_content():
    assert (
        env.normalize_prompt([{"role": "user", "content": "  "}, *PROMPT]) == PROMPT
    )


def test_completion_text_message_list():
    assert env.completion_text(completion(RIGHT)) == RIGHT


def test_completion_text_plain_string():
    assert env.completion_text(RIGHT) == RIGHT


def test_completion_text_empty():
    assert env.completion_text([]) == ""
    assert env.completion_text([{"role": "assistant", "content": None}]) == ""


def test_build_dataset_carries_grading_columns():
    ds = env.build_dataset(
        [
            {"prompt": PROMPT, "solution": GOLD},
            {"prompt": PROMPT, "solution": GOLD},
        ]
    )
    assert ds.column_names == ["prompt", "solution"]
    assert ds[0]["prompt"] == PROMPT
    assert ds["solution"] == [GOLD, GOLD]


def test_score_correct_boxed_answer():
    pytest.importorskip("math_verify")
    assert env.score_completion(completion(RIGHT), GOLD) == 1.0


def test_score_equivalent_form():
    pytest.importorskip("math_verify")
    assert env.score_completion(completion(r"\boxed{\frac{4}{6}}"), GOLD) == 1.0


def test_score_wrong_answer():
    pytest.importorskip("math_verify")
    assert env.score_completion(completion(WRONG), GOLD) == 0.0


def test_score_unparseable_answer_gets_nothing():
    pytest.importorskip("math_verify")
    assert env.score_completion(completion(NO_BOX), GOLD) == 0.0


def test_score_unparseable_gold_is_none():
    pytest.importorskip("math_verify")
    assert env.score_completion(completion(RIGHT), PROSE_GOLD) is None


def test_reward_fn_grades_a_group():
    pytest.importorskip("math_verify")
    reward_fn = env.make_reward_fn()
    assert reward_fn.__name__ == "deepmath"
    logged = {}
    rewards = reward_fn(
        prompts=["q", "q", "q"],
        completions=[completion(RIGHT), completion(WRONG), completion(NO_BOX)],
        solution=[GOLD] * 3,
        log_extra=lambda k, v: logged.setdefault(k, v),
    )
    assert rewards == [1.0, 0.0, 0.0]
    assert logged["solution"] == [GOLD] * 3
    assert len(logged["gold_parsed"]) == 3
    assert len(logged["answer_parsed"]) == 3


def test_score_without_math_verify_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", None)
    monkeypatch.setitem(sys.modules, "latex2sympy2_extended", None)
    with pytest.raises(ImportError, match="math-verify"):
        env.score_completion(completion(RIGHT), GOLD)


def fixture_parquet(path, rows):
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path)


def test_fetch_uses_filtered_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    fixture_parquet(
        cache / f"filtered-train-limit0-{env.REVISION[:8]}.parquet",
        [{"prompt": json.dumps(PROMPT), "solution": GOLD}],
    )
    tasks = env.fetch_tasks("train", str(cache), 0, env.REVISION, env.DATASET)
    assert tasks == [{"prompt": PROMPT, "solution": GOLD}]


def test_fetch_filters_and_caches(tmp_path, monkeypatch):
    pytest.importorskip("math_verify")
    cache = tmp_path / "cache"
    raw = tmp_path / "raw.parquet"
    fixture_parquet(
        raw,
        [
            {"prompt": PROMPT, "solution": GOLD},
            {"prompt": PROMPT, "solution": PROSE_GOLD},
            {"prompt": None, "solution": GOLD},
            {"prompt": PROMPT, "solution": "  "},
        ],
    )
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw.read_bytes())
        return dest

    monkeypatch.setattr(env, "_download", fake_download)
    tasks = env.fetch_tasks("train", str(cache), 0, env.REVISION, env.DATASET)
    assert tasks == [{"prompt": PROMPT, "solution": GOLD}]
    assert len(calls) == 1

    # Second run hits the filtered cache: no download at all.
    tasks = env.fetch_tasks("train", str(cache), 0, env.REVISION, env.DATASET)
    assert tasks == [{"prompt": PROMPT, "solution": GOLD}]
    assert len(calls) == 1


def test_deepmath_config_finalizes_and_matches_make():
    import sws

    from unsloth_agent.sws_utils import load_config_builder

    builder = load_config_builder("configs/grpo_deepmath.py")
    assert isinstance(builder, sws.Config)
    config, unused = builder.finalize([], return_unused_argv=True)
    assert unused == []
    assert config.exp_name == "qwen3-4b-grpo-deepmath"
    assert config.env_name == "deepmath"
    assert set(config.env.to_dict()) <= set(inspect.signature(env.make).parameters)
