# unsloth-agent

GRPO post-training with [Unsloth](https://unsloth.ai) + [TRL](https://huggingface.co/docs/trl) + [vLLM](https://docs.vllm.ai), configured with [sws](https://github.com/google/sws). Structure mirrors `~/personal/llm2`: typed `make(...)` factories, one file per type, sws configs merged from the CLI.

- `src/unsloth_agent/train/train_grpo.py` — loads base weights in bf16, attaches LoRA, trains with `GRPOTrainer` on the vLLM rollout engine (shared weights + standby mode), logs to trackio.
- `src/unsloth_agent/env/` — GRPO environments. `smoldataenvs` mimics the [SmolDataEnvs recipe](https://docs.while.ai/recipes/01-simulate/smol-data-envs): the policy writes a Python program, it runs next to the task's Kaggle tables, and the dataset's own grader (`grader.py`, vendored) scores the printed answer. Reward is 1/0; tasks whose tables fail to download are dropped at dataset build time (TRL needs dense rewards, so the recipe's "ungraded is `None`" becomes "drop the task"). `deepmath` mirrors the [TRL GRPO quickstart](https://huggingface.co/docs/trl/main/en/grpo_trainer) on `trl-lib/DeepMath-103K`: the policy answers a math problem in free text, and math-verify checks the `\boxed{}` answer against the gold symbolically (1/0; rows with unparseable gold are dropped at build, cached per revision).
- `configs/grpo_smoldataenvs.py` — full sws config. Override any value from the CLI: `... --config configs/grpo_smoldataenvs.py grpo.max_steps=10 env.limit=32`.
- `src/unsloth_agent/sws_utils.py` — config load/merge/CLI plumbing, copied from llm2.

## Tests (local, no GPU)

```bash
uv pip install --python .venv/bin/python pytest "sws-config>=0.6.3" datasets pandas pyarrow requests math-verify
.venv/bin/python -m pytest tests/ -q
```

(`math-verify` is the `deepmath` grader; without it the scoring tests skip.)

The full GPU stack (`unsloth`, `trl`, `vllm`) is only needed on the remote.

## Training (remote, see LSYNCD.md)

With `mycolab sync` running, install once per VM:

```bash
echo '!cd /content/unsloth-agent && pip install -e . --quiet' | colab exec
```

Smoke test (32 tasks, 10 steps), then the full run, both under tmux:

```bash
echo '!tmux new -d -s smoke "cd /content/unsloth-agent && PYTHONPATH=src python -m unsloth_agent.train.train_grpo --config configs/grpo_smoldataenvs.py env.limit=32 grpo.max_steps=10 grpo.save_steps=10 2>&1 | tee smoke.log"' | colab exec
echo '!tmux new -d -s train "cd /content/unsloth-agent && PYTHONPATH=src python -m unsloth_agent.train.train_grpo --config configs/grpo_smoldataenvs.py 2>&1 | tee train.log"' | colab exec
echo '!tmux capture-pane -p -t train | tail -20' | colab exec
```

Checkpoints land in `outputs/<project>/<exp>/`; env tables cache in `data/smoldataenvs/` (shared across runs, re-downloaded per VM). Metrics go to trackio (local-first); set `trackio.space_id=<user>/<space>` in the config or CLI to publish the dashboard to a Hugging Face Space.

Pull results back with `mycolab pull` (everything) or `colab download` (checkpoints only).
