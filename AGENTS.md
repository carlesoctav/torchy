# Agent Guidelines for Torchy

These rules and preferences must be strictly followed when working in this repository:

## 1. Tests & Scratch Testing
- **Never commit tests:** Do not create or commit test files in `tests/` or elsewhere unless explicitly requested by the user.
- **Use `temp/` for ad-hoc tests:** If quick scratch verification or testing is needed, put temporary scripts in `temp/`.
- **`temp/` is gitignored:** `temp/` is listed in `.gitignore` and must never be tracked in git.

## 2. Directory Structure & File Placement
- **Never put loose files in the outermost directory:** Keep the repository root clean.
  - Allowed in root: `pyproject.toml`, `README.md`, `AGENTS.md`, `.gitignore`, `uv.lock`, `.python-version`, and standard directory roots (`src/`, `configs/`, `scripts/`).
  - No loose scripts, test files, text dumps, or symlinks in the root directory.
- **Put scripts in `scripts/`:** Any standalone bash or python scripts belong strictly in `scripts/` (e.g., `scripts/run_vllm.sh`).
- **Code belongs in `src/torchy/`:** All library code, train modules, environments, patches, and benchmarks belong inside `src/torchy/` (e.g., `src/torchy/train/train_async_grpo.py`, `src/torchy/benchmark/trl_bench.py`).

## 3. Configuration Style
- **Never use `os.environ` in configs:** Configuration files in `configs/` must define clean, explicit Python defaults directly on `sws.Config()`. Never use `os.environ.get(...)`. Overrides should happen via CLI arguments or config inheritance.

## 4. Execution & Long-Running Commands
- **Use agent background capabilities:** When running long-running commands (such as `mycolab run` or servers), execute them as asynchronous/background tasks using agent tooling so the assistant never hangs waiting. Do not alter shell scripts with artificial `--background` flags.
- **No `lora_proxy`:** Always use the direct in-process patch `torchy.patch.wait_lora_patch` on `VLLMClient` instead of running a separate proxy process.

## 5. Security & Credentials
- **Never hardcode secrets or tokens in code:** Never embed API keys, write tokens, or credentials (such as Trackio tokens, HF tokens, etc.) directly in source files. Rely exclusively on external environment variables or configuration files.

## 6. Git & Remotes
- **Never push to remote:** Never run `git push` under any circumstances unless explicitly and directly instructed by the user. Keep all work strictly local.
