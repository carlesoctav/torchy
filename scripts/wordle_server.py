"""Self-hosted TextArena (Wordle) env server with a raised session cap.

The public server (https://openenv-wordle.hf.space) allows 1 session, while
one GRPO generation batch holds ``generation_batch_size`` concurrent
sessions. Run this next to training and point the config at it::

    python scripts/wordle_server.py
    WORDLE_MAX_SESSIONS=256 WORDLE_PORT=8001 python scripts/wordle_server.py

Honours the stock ``TEXTARENA_*`` vars (env id, players, turns). Restart to
drop sessions leaked by crashed runs (no idle reaping by default).
"""

from __future__ import annotations

import os

# Colab (and similar hosts) proxy all egress; NLTK's SSRF guard refuses
# proxied downloads unless told the proxy is trusted. This only fetches
# static public word lists, so opt in before the first download attempt.
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import uvicorn
from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.app import create_textarena_environment
from textarena_env.server.environment import TextArenaEnvironment

import random
from uuid import uuid4

import textarena as ta
from textarena.envs.Wordle.env import WordleEnv

# The stock server caps at 1 session and TextArena doesn't declare
# concurrent-session support. Each session is an independent game instance
# (own state, own executor), so this is safe to flip.
TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True

# Patch WordleEnv.reset to ensure deterministic target word selection per seed
# using an isolated Random generator (thread-safe and immune to global PRNG races).
def _wordle_reset(self, num_players: int = 1, seed: int | None = None):
    self.state = ta.SinglePlayerState(num_players=num_players, seed=seed)
    rng = random.Random(seed) if seed is not None else random
    game_state = {
        "secret_word": rng.choice(self.word_list),
        "guess_history": [],
        "word_length": self.word_length,
        "num_guesses": self.num_guesses,
    }
    self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

WordleEnv.reset = _wordle_reset

# Patch TextArenaEnvironment.reset to forward seed to self._ta_env.reset
def _env_reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs):
    env = self._ta_env
    while hasattr(env, "env"):
        if hasattr(env, "full_observations"):
            env.full_observations = {}
        env = env.env
    if hasattr(env, "full_observations"):
        env.full_observations = {}

    self._ta_env.reset(num_players=self.num_players, seed=seed)

    for provider in self._reward_providers:
        provider.reset()

    self._state.episode_id = episode_id if episode_id is not None else str(uuid4())
    self._state.step_count = 0
    self._state.turn = 0
    self._state.last_reward = 0.0
    self._state.last_info = {}
    self._state.raw_state = self._snapshot_state()
    self._last_reward_signals = {}

    observation = self._build_observation()
    observation.reward = 0.0
    observation.done = False

    return observation

TextArenaEnvironment.reset = _env_reset

app = create_app(
    create_textarena_environment,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
    max_concurrent_envs=int(os.getenv("WORDLE_MAX_SESSIONS", "128")),
)


def main() -> None:
    uvicorn.run(
        app,
        host=os.getenv("WORDLE_HOST", "0.0.0.0"),
        port=int(os.getenv("WORDLE_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
