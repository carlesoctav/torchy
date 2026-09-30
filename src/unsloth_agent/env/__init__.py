import importlib


def make_env(env_name: str, env_config: dict):
    env_module = importlib.import_module(f"unsloth_agent.env.{env_name}")
    return env_module.make(**env_config)
