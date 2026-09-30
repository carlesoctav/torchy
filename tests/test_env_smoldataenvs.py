"""Offline tests for the SmolDataEnvs env: no network, no GPU.

Tables are tiny CSVs in a tmp dir; the graded programs only use the stdlib.
"""

import pytest

from unsloth_agent.env import smoldataenvs as env

SURVEY_CSV = "age,employed\n25,yes\n34,yes\n41,no\n55,yes\n"


@pytest.fixture()
def table_dir(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "survey.csv").write_text(SURVEY_CSV)
    return str(tables)


def task(table_dir, **overrides):
    row = {
        "task_id": "t1",
        "question": "How many respondents are older than 30 and employed?",
        "answer": "2",
        "files": ["survey.csv"],
        "reward_mode": "numeric",
        "atol": 0,
        "rtol": 0.0,
        "table_dir": table_dir,
    }
    row.update(overrides)
    return row


def fence(code, lang="python"):
    return f"```{lang}\n{code}\n```"


GOOD_PROGRAM = """
import csv
rows = list(csv.DictReader(open("input/survey.csv")))
print(sum(1 for r in rows if int(r["age"]) > 30 and r["employed"] == "yes"))
"""


def test_prompt_lists_files_and_question(table_dir):
    prompt = env.prompt_for(task(table_dir))
    assert "input/survey.csv" in prompt
    assert "older than 30" in prompt


def test_split_answer_takes_last_fenced_block():
    text = fence("print(1)", "python") + "\n" + fence("echo hi", "bash")
    assert env.split_answer(text) == ("bash", "echo hi")


def test_split_answer_without_fence_returns_whole_text():
    assert env.split_answer("  42 ") == ("", "42")


def test_run_program_prints_last_line(table_dir):
    printed, why = env.run_program(GOOD_PROGRAM, table_dir, ["survey.csv"])
    assert (printed, why) == ("2", "ran")


def test_run_program_reports_crash(table_dir):
    printed, why = env.run_program("raise KeyError('age')", table_dir, ["survey.csv"])
    assert printed is None
    assert "KeyError" in why


def test_run_program_reports_timeout(table_dir):
    printed, why = env.run_program(
        "import time; time.sleep(30)", table_dir, ["survey.csv"], timeout=1
    )
    assert printed is None
    assert "timed out" in why


def test_run_program_reports_empty_output(table_dir):
    printed, why = env.run_program("x = 1", table_dir, ["survey.csv"])
    assert printed is None
    assert "printed nothing" in why


def test_score_correct_program(table_dir):
    row = task(table_dir)
    assert (
        env.score_completion(
            fence(GOOD_PROGRAM),
            row["answer"],
            row["files"],
            row["table_dir"],
            row["reward_mode"],
            row["atol"],
            row["rtol"],
        )
        == 1.0
    )


def test_score_wrong_value(table_dir):
    row = task(table_dir)
    assert (
        env.score_completion(
            fence("print(99)"),
            row["answer"],
            row["files"],
            row["table_dir"],
            row["reward_mode"],
            row["atol"],
            row["rtol"],
        )
        == 0.0
    )


def test_score_shell_answer_gets_nothing(table_dir):
    row = task(table_dir)
    assert (
        env.score_completion(
            fence('echo "2"', "bash"),
            row["answer"],
            row["files"],
            row["table_dir"],
            row["reward_mode"],
            row["atol"],
            row["rtol"],
        )
        == 0.0
    )


def test_score_crash_gets_nothing(table_dir):
    row = task(table_dir)
    assert (
        env.score_completion(
            fence("raise KeyError('age')"),
            row["answer"],
            row["files"],
            row["table_dir"],
            row["reward_mode"],
            row["atol"],
            row["rtol"],
        )
        == 0.0
    )


def test_score_honors_tolerance(table_dir):
    row = task(table_dir, answer="5768.04", atol=0.05, rtol=0.01)
    assert (
        env.score_completion(
            fence("print(5768)"),
            row["answer"],
            row["files"],
            row["table_dir"],
            row["reward_mode"],
            row["atol"],
            row["rtol"],
        )
        == 1.0
    )


def test_build_dataset_carries_grading_columns(table_dir):
    ds = env.build_dataset([task(table_dir), task(table_dir, task_id="t2")])
    assert ds.column_names == [
        "prompt",
        "task_id",
        "reference",
        "files",
        "table_dir",
        "reward_mode",
        "atol",
        "rtol",
    ]
    assert ds[0]["prompt"] == [
        {"role": "system", "content": env.SYSTEM},
        {"role": "user", "content": env.prompt_for(task(table_dir))},
    ]
    assert ds["reference"] == ["2", "2"]


def test_reward_fn_grades_a_group(table_dir):
    row = task(table_dir)
    reward_fn = env.make_reward_fn(timeout=30.0, num_workers=2)
    assert reward_fn.__name__ == "smoldataenvs"
    rewards = reward_fn(
        prompts=["q", "q"],
        completions=[fence(GOOD_PROGRAM), fence("print(99)")],
        reference=[row["answer"]] * 2,
        files=[row["files"]] * 2,
        table_dir=[row["table_dir"]] * 2,
        reward_mode=[row["reward_mode"]] * 2,
        atol=[row["atol"]] * 2,
        rtol=[row["rtol"]] * 2,
    )
    assert rewards == [1.0, 0.0]
