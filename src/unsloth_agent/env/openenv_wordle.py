from __future__ import annotations

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


def make(
    prompt: str = PROMPT, dataset_size: int = 1000, env_url: str = ENV_URL
) -> dict[str, Any]:
    def reward_fn(environments, **kwargs):
        return [env.reward for env in environments]

    reward_fn.__name__ = "openenv_wordle"
    EnvFactory = partial(WordleEnv, env_url = env_url)

    return {
        "train_dataset": Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": prompt}] for _ in range(dataset_size)
                ]
            }
        ),
        "reward_funcs": [reward_fn],
        "environment_factory": EnvFactory,
    }
