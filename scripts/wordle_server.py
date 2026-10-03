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

# The stock server caps at 1 session and TextArena doesn't declare
# concurrent-session support. Each session is an independent game instance
# (own state, own executor), so this is safe to flip.
TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True

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
