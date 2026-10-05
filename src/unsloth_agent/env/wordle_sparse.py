from __future__ import annotations

import random
import re
from functools import partial
from typing import Any

from datasets import Dataset
from textarena_env import TextArenaAction, TextArenaEnv

#: Hosted TextArena server.
ENV_URL = "https://openenv-wordle.hf.space"
PROMPT = """
You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.
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


class WordleSparseEnv:
    """One Wordle game per rollout with sparse outcome reward and duplicate/invalid penalties."""

    INVALID_MARKERS = [
        "You have already guessed",
        "You attempted an invalid move",
        "wrong format",
        "is not an English word",
        "must be exactly",
    ]

    def __init__(self, env_url: str):
        # The client is async but TRL's reset() path is sync.
        self.client = TextArenaEnv(base_url=env_url).sync()
        self.reward = 0.0
        self.done = False
        self.turns = 0
        self.invalid_count = 0
        self.guessed_words: set[str] = set()
        self._last_full_feedback = ""

    def reset(self, **kwargs) -> str | None:
        seed = kwargs.get("seed")
        result = self.client.reset(seed=seed) if seed is not None else self.client.reset()
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        self.turns = 0
        self.invalid_count = 0
        self.guessed_words.clear()
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

        self.turns += 1
        guess_str = str(guess).strip()

        # Check for duplicate word or formatting errors locally
        match = re.search(r"\[([A-Za-z]+)\]", guess_str)
        is_invalid = False
        if match:
            word = match.group(1).lower()
            if word in self.guessed_words:
                is_invalid = True
            self.guessed_words.add(word)
        else:
            is_invalid = True

        result = self.client.step(TextArenaAction(message=guess_str))
        full_feedback = result.observation.messages[0].content
        feedback = full_feedback[len(self._last_full_feedback) :]
        self._last_full_feedback = full_feedback

        # Also inspect server feedback for invalid move warnings
        if any(marker in feedback for marker in self.INVALID_MARKERS):
            is_invalid = True

        if is_invalid:
            self.invalid_count += 1

        self.done = result.done

        # Reward calculation:
        # TextArena sets reward == 1.0 (or "Congratulations") only when won.
        won = (result.reward == 1.0) or ("Congratulations" in full_feedback)

        penalty = 0.25 * self.invalid_count
        if self.done:
            if won:
                # Speed bonus: 0.1 for every unused turn remaining (up to +0.5 for a 1-turn win)
                speed_bonus = max(0, 6 - self.turns) * 0.1
                # Base win reward = 1.0, minimum floor = 0.2 even with prior mistakes
                self.reward = max(0.2, 1.0 + speed_bonus - penalty)
            else:
                # Loss yields 0.0 minus penalties for invalid/duplicate attempts
                self.reward = 0.0 - penalty
        else:
            # Intermediate step: track accumulated penalty
            self.reward = 0.0 - penalty

        return feedback

    def get_reward(self) -> float:
        """Return the episode reward for TRL environment reward tracking."""
        return self.reward


def make(
    prompt: str = PROMPT,
    dataset_size: int = 1000,
    env_url: str = ENV_URL,
    seed: int | None = 42,
) -> dict[str, Any]:
    def reward_fn(environments, **kwargs):
        return [env.reward for env in environments]

    reward_fn.__name__ = "wordle_sparse"
    EnvFactory = partial(WordleSparseEnv, env_url=env_url)

    rng = random.Random(seed) if seed is not None else random
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(dataset_size)]

    return {
        "train_dataset": Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": prompt}] for _ in range(dataset_size)
                ],
                "seed": seeds,
            }
        ),
        "reward_funcs": [reward_fn],
        "environment_factory": EnvFactory,
    }
