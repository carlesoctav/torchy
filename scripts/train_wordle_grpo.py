"""Standalone Wordle GRPO: no Unsloth, no sws, TRL-native LoRA.

Based on TRL's ``examples/grpo_multi_env`` (Wordle half only), with the
hyperparameters from ``configs/wordle.py``. Everything tunable lives in the
block below; the rest is wiring.

Needs the env server (``scripts/wordle_server.py``) reachable at ``ENV_URL``,
and ``pip install "openenv-textarena @
git+https://huggingface.co/spaces/openenv/wordle"``.

Run (from the repo root, GPU box):
    uv run python scripts/train_wordle_grpo.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Knobs: change stuff here.
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-1.7B"
ENV_URL = "http://localhost:8001"
DATASET_SIZE = 1000
OUTPUT_DIR = "outputs/wordle-qwen3/qwen3-1.7b-grpo-wordle-standalone"
SEED = 3407

PER_DEVICE_BATCH = 4
GRAD_ACCUM = 1
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512
MAX_STEPS = 3000
SAVE_STEPS = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0.05
LR_SCHEDULER = "constant"
OPTIM = "adamw_8bit"
MAX_GRAD_NORM = 1.0
BETA = 0.001
ENABLE_THINKING = False

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

USE_VLLM = True  # configs/wordle.py has vLLM commented out; flip to try it
REPORT_TO = "trackio"
vllm_gpu_memory_utilization = 0.3
vllm_max_model_length = 4096
# ---------------------------------------------------------------------------

import sys
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from textarena_env import TextArenaAction, TextArenaEnv
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


PROMPT = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""


class WordleEnv:
    """One Wordle game per rollout; ``reset`` opens it, ``guess`` plays it."""

    def __init__(self, env_url):
        # The client is async but TRL's reset() path is sync.
        self.client = TextArenaEnv(base_url=env_url).sync()
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset()
        # TextArena returns the full game history every turn; keep it so
        # guess() can slice out just the new feedback.
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        full_feedback = result.observation.messages[0].content
        feedback = full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = full_feedback
        # Invalid moves keep the last reward server-side; zero them here.
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback


def wordle_reward(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(DATASET_SIZE)]}
    )

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=Path(OUTPUT_DIR).name,
        seed=SEED,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing = True,
        use_liger_kernel = True,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        beta=BETA,
        use_vllm=False,
        vllm_max_model_length = 4096,
        vllm_gpu_memory_utilization = 0.3,
        report_to=REPORT_TO,
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=wordle_reward,
        train_dataset=dataset,
        args=args,
        environment_factory=partial(WordleEnv, env_url=ENV_URL),
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    sys.exit(main())
