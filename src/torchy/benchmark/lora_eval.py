from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import requests
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.chat_template_utils import add_response_schema, parse_response
from trl.experimental.async_grpo.vllm_client import VLLMClient

# Ensure repo root and src/ are importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import sws
from torchy.env import make_env
from torchy.patch import wait_lora_patch
from torchy.sws_utils import run as sws_run

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("LoraEval")


def discover_checkpoints(
    checkpoints_dir: str | Path | None = None,
    explicit_checkpoints: Sequence[str | Path] | None = None,
    eval_base_model: bool = False,
    base_model_name: str = "base_model",
) -> list[tuple[str, Path | None, int]]:
    """Discover and order checkpoints to evaluate.

    Returns:
        List of tuples: (checkpoint_display_name, checkpoint_path_or_None, step_int)
        If checkpoint_path is None, it represents the unadapted base model.
    """
    discovered: list[tuple[str, Path | None, int]] = []

    if eval_base_model:
        discovered.append((base_model_name, None, 0))

    if explicit_checkpoints:
        for item in explicit_checkpoints:
            p = Path(item)
            name = p.name
            m = re.search(r"checkpoint-(\d+)", name)
            step = int(m.group(1)) if m else (999999 if "final" in name else 0)
            discovered.append((name, p, step))
        return discovered

    if not checkpoints_dir:
        return discovered

    root = Path(checkpoints_dir)
    if not root.exists():
        logger.info(
            f"Checkpoints directory {root} not visible locally (using remote paths for vLLM)."
        )
        subdirs = []
        standard_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        for s in standard_steps:
            subdirs.append((f"checkpoint-{s}", root / f"checkpoint-{s}", s))
        subdirs.append(("final-adapter", root / "final-adapter", 999999))
        discovered.extend(subdirs)
        return discovered

    # Check if root itself is a checkpoint
    if (root / "adapter_config.json").exists() or (root / "adapter_model.safetensors").exists():
        name = root.name
        m = re.search(r"checkpoint-(\d+)", name)
        step = int(m.group(1)) if m else (999999 if "final" in name else 0)
        discovered.append((name, root, step))
        return discovered

    # Scan subdirectories
    subdirs: list[tuple[str, Path, int]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        has_adapter = (
            (entry / "adapter_config.json").exists()
            or (entry / "adapter_model.safetensors").exists()
            or entry.name.startswith("checkpoint-")
            or "final-adapter" in entry.name
        )
        if not has_adapter:
            continue

        name = entry.name
        m = re.search(r"checkpoint-(\d+)", name)
        if m:
            step = int(m.group(1))
        elif "final" in name:
            step = 999999
        else:
            step = 0
        subdirs.append((name, entry, step))

    subdirs.sort(key=lambda x: (x[2], x[0]))
    discovered.extend(subdirs)
    return discovered


class BenchmarkEnv:
    """Benchmark evaluation runner talking directly to vLLM.

    Mimics AsyncGRPOTrainer / AsyncRolloutWorker generation and scoring loop,
    handling both dataset-based reward function environments (e.g. sanity_math)
    and interactive tool environments (e.g. openenv_wordle).
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Math-1.5B",
        tokenizer: PreTrainedTokenizerBase | None = None,
        vllm_server_base_url: str = "http://localhost:8000",
        max_completion_length: int = 3000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tool_calling_iterations: int = 10,
        concurrency: int = 8,
        log_completions: bool = True,
        num_completions_to_print: int = 5,
        output_dir: str | Path = "outputs/bench",
        **kwargs,
    ):
        self.base_model = base_model
        self.vllm_server_base_url = vllm_server_base_url.rstrip("/")
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.chat_template_kwargs = (
            copy.deepcopy(chat_template_kwargs)
            if chat_template_kwargs is not None
            else {"enable_thinking": True}
        )
        self.max_tool_calling_iterations = max_tool_calling_iterations
        self.concurrency = concurrency
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print
        self.output_dir = Path(output_dir)
        self.max_model_len = 4096
        self.extra_kwargs = kwargs

        if tokenizer is None:
            logger.info(f"Loading tokenizer for {self.base_model}...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        has_template = getattr(tokenizer, "response_template", None) is not None
        has_schema = getattr(tokenizer, "response_schema", None) is not None
        if not has_template and not has_schema:
            try:
                tokenizer = add_response_schema(tokenizer)
            except Exception as e:
                logger.warning(f"Could not apply add_response_schema: {e}")
        self.tokenizer = tokenizer

    def wait_for_vllm(self, timeout: float = 300.0, poll_interval_s: float = 2.0) -> None:
        """Wait until vLLM server is healthy and responding."""
        logger.info(f"Waiting for vLLM server at {self.vllm_server_base_url} ...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = requests.get(f"{self.vllm_server_base_url}/health", timeout=5)
                if res.status_code == 200:
                    try:
                        m_res = requests.get(f"{self.vllm_server_base_url}/v1/models", timeout=5)
                        if m_res.status_code == 200:
                            data = m_res.json()
                            if data.get("data"):
                                self.max_model_len = data["data"][0].get("max_model_len", self.max_model_len)
                            logger.info(
                                f"vLLM server ready ({time.time() - start:.1f}s), max_model_len={self.max_model_len}"
                            )
                            return
                    except Exception:
                        pass
                    logger.info(f"vLLM server ready ({time.time() - start:.1f}s)")
                    return
            except Exception:
                pass
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"vLLM server at {self.vllm_server_base_url} did not become ready within {timeout}s."
        )

    async def _generate_one_turn(
        self,
        session: aiohttp.ClientSession,
        prompt_ids: list[int],
        model_name: str,
    ) -> tuple[list[int] | None, str]:
        """Request a single turn completion from vLLM for the given model/LoRA adapter name."""
        prompt_len = len(prompt_ids)
        available_tokens = self.max_model_len - prompt_len
        if available_tokens <= 16:
            logger.warning(
                f"Prompt length {prompt_len} reached context limit {self.max_model_len}; stopping turn generation."
            )
            return None, ""

        max_tokens = min(self.max_completion_length, available_tokens)
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "return_token_ids": True,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty

        url = f"{self.vllm_server_base_url}/v1/completions"
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=240)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choice = data["choices"][0]
                        turn_text = choice.get("text", "")
                        turn_ids = choice.get("token_ids")
                        if turn_ids is None:
                            turn_ids = self.tokenizer.encode(turn_text, add_special_tokens=False)
                        return turn_ids, turn_text
                    else:
                        err_text = await resp.text()
                        logger.warning(f"vLLM status {resp.status} (model={model_name}): {err_text}")
            except Exception as e:
                logger.warning(f"vLLM request error on attempt {attempt + 1}: {e}")
            await asyncio.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Failed to generate completion from vLLM after {max_attempts} attempts (model={model_name})."
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_dict: dict[str, Callable],
    ) -> tuple[list[dict[str, str]], int, int, int]:
        loop = asyncio.get_running_loop()
        tool_messages = []
        n_calls = 0
        n_failures = 0
        n_invalid = 0

        for tool_call in tool_calls:
            n_calls += 1
            func_data = tool_call.get("function", {})
            name = func_data.get("name")
            call_id = tool_call.get("id")
            tool = tool_dict.get(name)

            if tool is None:
                n_failures += 1
                msg = {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps({"error": f"Unknown tool '{name}'"}),
                }
                if call_id:
                    msg["tool_call_id"] = call_id
                tool_messages.append(msg)
                continue

            raw_args = func_data.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = raw_args
            else:
                arguments = raw_args

            try:
                if inspect.iscoroutinefunction(tool):
                    if isinstance(arguments, dict):
                        result = await tool(**arguments)
                    else:
                        result = await tool(arguments)
                else:
                    if isinstance(arguments, dict):
                        try:
                            result = await loop.run_in_executor(
                                None, functools.partial(tool, **arguments)
                            )
                        except TypeError:
                            first_val = list(arguments.values())[0] if arguments else ""
                            result = await loop.run_in_executor(
                                None, functools.partial(tool, first_val)
                            )
                    else:
                        result = await loop.run_in_executor(
                            None, functools.partial(tool, arguments)
                        )
            except Exception as e:
                n_failures += 1
                result = str({"error": str(e)})

            result_str = str(result)
            if "invalid move" in result_str.lower() or "wrong format" in result_str.lower():
                n_invalid += 1

            msg = {"role": "tool", "name": name, "content": result_str}
            if call_id:
                msg["tool_call_id"] = call_id
            tool_messages.append(msg)

        return tool_messages, n_calls, n_failures, n_invalid

    def _print_episode_trace(
        self,
        idx: int,
        record: dict[str, Any],
        prompt_messages: list[dict[str, Any]],
        trace_log: list[dict[str, Any]],
    ) -> None:
        outcome = "WON / CORRECT" if record["won"] else "LOST / INCORRECT"
        print("\n" + "=" * 80)
        print(
            f"[Episode {idx + 1}] Model: {record['model_name']} | Outcome: {outcome} | "
            f"Reward: {record['reward']:.2f} | Turns: {record['turns']} | Time: {record['duration_s']}s"
        )
        if record.get("solution"):
            print(f"Ground Truth Solution: {record['solution']}")
        print("-" * 80)
        for t in trace_log:
            turn_num = t.get("turn")
            print(f"Turn {turn_num}:")
            if t.get("reasoning"):
                print(f"  [Thinking]: {t['reasoning'].strip()}")
            if t.get("content"):
                print(f"  [Content]: {t['content'].strip()}")
            if t.get("tool_calls"):
                for tc in t["tool_calls"]:
                    fn = tc.get("function", {})
                    print(f"  [Tool Call]: {fn.get('name')}({fn.get('arguments')})")
            if t.get("tool_responses"):
                for tr in t["tool_responses"]:
                    print(f"  [Tool Response]: {tr.strip()}")
        print("-" * 80)
        print(f"Result: {outcome} (Reward: {record['reward']:.2f})")
        print("=" * 80 + "\n")

    async def _eval_one_episode(
        self,
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        idx: int,
        row: dict[str, Any],
        model_name: str,
        checkpoint_name: str,
        step: int,
        env_factory: Callable | None,
        env_pool: list[Any],
        reward_funcs: Any | None,
    ) -> dict[str, Any]:
        async with sem:
            t0 = time.time()
            env = None
            env_methods = []
            tool_dict = {}

            if env_factory is not None:
                env = env_pool.pop() if env_pool else env_factory()
                reset_kwargs = {
                    k: v for k, v in row.items() if k not in ("prompt", "solution", "environment")
                }
                try:
                    observation = env.reset(**reset_kwargs)
                except Exception as e:
                    logger.error(f"[Episode {idx}] env.reset failed: {e}")
                    observation = None

                env_methods = [
                    member
                    for member_name, member in inspect.getmembers(
                        env, predicate=inspect.ismethod
                    )
                    if member_name not in ("reset", "get_reward")
                    and not member_name.startswith("_")
                ]
                for m in env_methods:
                    if not getattr(m, "__doc__", None):
                        m.__func__.__doc__ = f"Call {m.__name__}."
                tool_dict = {m.__name__: m for m in env_methods}
            else:
                observation = None

            # Prepare initial prompt messages
            if "prompt" in row:
                raw_prompt = row["prompt"]
                if isinstance(raw_prompt, str):
                    prompt_messages = [{"role": "user", "content": raw_prompt}]
                elif isinstance(raw_prompt, list):
                    prompt_messages = copy.deepcopy(raw_prompt)
                else:
                    prompt_messages = [{"role": "user", "content": str(raw_prompt)}]
            else:
                prompt_messages = [{"role": "user", "content": ""}]

            if observation is not None and prompt_messages:
                last_msg = copy.deepcopy(prompt_messages[-1])
                content = last_msg.get("content", "")
                if isinstance(observation, list) and isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(observation, str) and isinstance(content, list):
                    observation = [{"type": "text", "text": observation}]
                last_msg["content"] = content + observation
                prompt_messages = prompt_messages[:-1] + [last_msg]

            messages = copy.deepcopy(prompt_messages)

            tool_calls_count = 0
            tool_failures_count = 0
            invalid_moves_count = 0
            turns = 0
            all_completion_ids: list[int] = []
            trace_log = []
            last_turn_text = ""

            while True:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_dict=False,
                    add_generation_prompt=True,
                    tools=env_methods or None,
                    **self.chat_template_kwargs,
                )

                turn_ids, turn_text = await self._generate_one_turn(
                    session, prompt_ids, model_name=model_name
                )
                last_turn_text = turn_text
                if turn_ids is None:
                    trace_log.append({
                        "turn": turns + 1,
                        "reasoning": "",
                        "content": "Context limit reached.",
                        "stop_reason": "context_limit",
                    })
                    break

                all_completion_ids.extend(turn_ids)
                try:
                    assistant_message = parse_response(
                        self.tokenizer, turn_ids, prefix=prompt_ids
                    )
                except Exception:
                    content_clean = turn_text
                    reasoning_clean = ""
                    think_match = re.search(r"<think>(.*?)</think>", turn_text, re.DOTALL)
                    if think_match:
                        reasoning_clean = think_match.group(1).strip()
                        content_clean = turn_text[think_match.end():].strip()
                    assistant_message = {
                        "role": "assistant",
                        "content": content_clean,
                        "reasoning_content": reasoning_clean,
                    }

                # Fallback parser if XML-style tool call is inside content
                if not assistant_message.get("tool_calls") and "<tool_call>" in turn_text:
                    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", turn_text, re.DOTALL)
                    if m:
                        try:
                            parsed_tc = json.loads(m.group(1))
                            if "name" in parsed_tc:
                                assistant_message["tool_calls"] = [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": parsed_tc["name"],
                                            "arguments": parsed_tc.get("arguments", {}),
                                        },
                                    }
                                ]
                        except Exception:
                            pass

                messages.append(assistant_message)
                turns += 1

                reasoning = assistant_message.get("reasoning_content", "")
                content_text = assistant_message.get("content", "")
                tool_calls = assistant_message.get("tool_calls")

                turn_trace = {
                    "turn": turns,
                    "reasoning": reasoning,
                    "content": content_text,
                }
                if tool_calls:
                    turn_trace["tool_calls"] = tool_calls

                if not tool_calls:
                    trace_log.append(turn_trace)
                    break

                if turns >= self.max_tool_calling_iterations:
                    turn_trace["stop_reason"] = "max_iterations"
                    trace_log.append(turn_trace)
                    break

                tool_msgs, n_calls, n_fails, n_inv = await self._execute_tool_calls(
                    tool_calls, tool_dict
                )
                tool_calls_count += n_calls
                tool_failures_count += n_fails
                invalid_moves_count += n_inv
                messages.extend(tool_msgs)

                turn_trace["tool_responses"] = [m["content"] for m in tool_msgs]
                trace_log.append(turn_trace)

                if env is not None and getattr(env, "done", False):
                    break

            # Compute reward
            reward = 0.0
            if reward_funcs is not None:
                try:
                    assistant_completions = [m for m in messages if m.get("role") == "assistant"]
                    # Format matching TRL AsyncRolloutWorker completions input
                    completions_arg = [assistant_completions if assistant_completions else [{"role": "assistant", "content": last_turn_text}]]
                    solution_arg = [row.get("solution", "")]
                    ids_arg = [all_completion_ids]

                    rf = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
                    rf_scores = []
                    for fn in rf:
                        try:
                            scores = fn(
                                completions=completions_arg,
                                solution=solution_arg,
                                completion_ids=ids_arg,
                                prompts=[prompt_messages],
                            )
                            if inspect.iscoroutine(scores):
                                scores = await scores
                            rf_scores.append(float(scores[0]) if scores else 0.0)
                        except TypeError:
                            # Fallback if reward func accepts alternative kwargs
                            scores = fn(
                                completions=completions_arg,
                                solution=solution_arg,
                                completion_ids=ids_arg,
                            )
                            if inspect.iscoroutine(scores):
                                scores = await scores
                            rf_scores.append(float(scores[0]) if scores else 0.0)
                    reward = float(np.mean(rf_scores)) if rf_scores else 0.0
                except Exception as e:
                    logger.warning(f"[Episode {idx}] Error evaluating reward_funcs: {e}")
                    reward = 0.0
            elif env is not None:
                if hasattr(env, "get_reward"):
                    reward = env.get_reward()
                    if inspect.iscoroutine(reward):
                        reward = await reward
                elif hasattr(env, "reward"):
                    reward = env.reward
                reward = float(reward)

            won = (reward == 1.0) or any(
                isinstance(m.get("content"), str) and "Congratulations" in m["content"]
                for m in messages
            )

            if env is not None:
                env_pool.append(env)

            duration_s = time.time() - t0

            try:
                rendered_trace = self.tokenizer.apply_chat_template(
                    messages,
                    tools=env_methods or None,
                    tokenize=False,
                    **self.chat_template_kwargs,
                )
            except Exception:
                rendered_trace = ""

            episode_record = {
                "episode_idx": idx,
                "checkpoint": checkpoint_name,
                "step": step,
                "model_name": model_name,
                "seed": row.get("seed"),
                "solution": row.get("solution"),
                "won": bool(won),
                "reward": float(reward),
                "turns": turns,
                "completion_tokens": len(all_completion_ids),
                "tool_calls_count": tool_calls_count,
                "tool_failures_count": tool_failures_count,
                "invalid_moves_count": invalid_moves_count,
                "duration_s": round(duration_s, 2),
                "prompt_json": json.dumps(prompt_messages),
                "messages_json": json.dumps(messages),
                "trace_json": json.dumps(trace_log),
                "rendered_trace": rendered_trace,
            }

            if self.log_completions and (idx < self.num_completions_to_print):
                self._print_episode_trace(idx, episode_record, prompt_messages, trace_log)
            else:
                logger.info(
                    f"[{checkpoint_name} (step {step})] Ep {idx + 1} | "
                    f"Outcome: {'CORRECT' if won else 'INCORRECT'} | Reward: {reward:.2f} | "
                    f"Tokens: {len(all_completion_ids)} | Time: {duration_s:.1f}s"
                )

            return episode_record

    async def eval_checkpoint(
        self,
        dataset: Dataset | Sequence[dict[str, Any]],
        model_name: str,
        checkpoint_name: str,
        step: int,
        env_factory: Callable | None = None,
        reward_funcs: Any | None = None,
        save_parquet: bool = True,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Evaluate one model / LoRA checkpoint on the dataset."""
        self.wait_for_vllm()

        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        elif hasattr(dataset, "__iter__"):
            rows = [dataset[i] for i in range(len(dataset))]
        else:
            rows = list(dataset)

        total_episodes = len(rows)
        logger.info(
            f"Evaluating checkpoint '{checkpoint_name}' (step {step}) on {total_episodes} cases "
            f"via model '{model_name}' (concurrency={self.concurrency})"
        )

        sem = asyncio.Semaphore(self.concurrency)
        env_pool = [env_factory() for _ in range(self.concurrency)] if env_factory else []

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.concurrency * 2)
        ) as session:
            tasks = [
                asyncio.create_task(
                    self._eval_one_episode(
                        sem=sem,
                        session=session,
                        idx=i,
                        row=row,
                        model_name=model_name,
                        checkpoint_name=checkpoint_name,
                        step=step,
                        env_factory=env_factory,
                        env_pool=env_pool,
                        reward_funcs=reward_funcs,
                    )
                )
                for i, row in enumerate(rows)
            ]
            records = await asyncio.gather(*tasks)

        df = pd.DataFrame(records)

        rewards = df["reward"].values if not df.empty else np.array([])
        mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        std_reward = float(np.std(rewards)) if len(rewards) > 0 else 0.0
        sem_reward = float(std_reward / np.sqrt(len(rewards))) if len(rewards) > 0 else 0.0
        min_reward = float(np.min(rewards)) if len(rewards) > 0 else 0.0
        max_reward = float(np.max(rewards)) if len(rewards) > 0 else 0.0
        total_wins = int(df["won"].sum()) if not df.empty else 0
        win_rate = float(total_wins / total_episodes) if total_episodes > 0 else 0.0
        mean_turns = float(df["turns"].mean()) if not df.empty else 0.0
        mean_tokens = float(df["completion_tokens"].mean()) if not df.empty else 0.0
        mean_duration = float(df["duration_s"].mean()) if not df.empty else 0.0

        metrics = {
            "checkpoint_name": checkpoint_name,
            "step": step,
            "model_name": model_name,
            "total_episodes": total_episodes,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "sem_reward": sem_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "wins": total_wins,
            "win_rate": win_rate,
            "mean_turns": mean_turns,
            "mean_completion_tokens": mean_tokens,
            "mean_duration_s": mean_duration,
        }

        if save_parquet:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            clean_name = re.sub(r"[^\w\-_\.]", "_", checkpoint_name)
            parquet_path = self.output_dir / f"{clean_name}.parquet"
            df.to_parquet(parquet_path, index=False)
            metrics["parquet_path"] = str(parquet_path)
            logger.info(f"Saved traces to {parquet_path}")

        print("\n" + "-" * 70)
        print(f"CHECKPOINT EVALUATION: {checkpoint_name} (step {step})")
        print(f"Mean Reward:   {mean_reward:.4f} ± {sem_reward:.4f} (std: {std_reward:.4f})")
        print(f"Win Rate:      {win_rate * 100:.2f}% ({total_wins}/{total_episodes})")
        print(f"Mean Tokens:   {mean_tokens:.1f}")
        print(f"Mean Duration: {mean_duration:.2f}s")
        print("-" * 70 + "\n")

        return metrics, df


class LoraBenchmarkRunner:
    """Orchestrates multi-checkpoint LoRA evaluation against vLLM.

    Applies wait_lora_patch, discovers all checkpoints, hot-swaps each adapter
    in vLLM, executes BenchmarkEnv.eval_checkpoint(), unloads adapter, and
    compiles a comparative summary report.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-Math-1.5B",
        vllm_server_base_url: str = "http://localhost:8000",
        checkpoints_dir: str | Path | None = None,
        explicit_checkpoints: Sequence[str | Path] | None = None,
        eval_base_model: bool = True,
        output_dir: str | Path = "outputs/bench",
        benchmark_env: BenchmarkEnv | None = None,
        report_to: str | None = None,
        project_name: str = "lora-bench",
        exp_name: str = "eval",
        run_name: str | None = None,
        overwrite: bool = False,
        **kwargs,
    ):
        self.base_model_name = base_model_name
        self.vllm_server_base_url = vllm_server_base_url.rstrip("/")
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self.explicit_checkpoints = explicit_checkpoints
        self.eval_base_model = eval_base_model
        self.output_dir = Path(output_dir)
        self.report_to = report_to
        self.project_name = project_name
        self.exp_name = exp_name
        self.run_name = run_name or f"{project_name}_{exp_name}"
        self.overwrite = overwrite

        # Ensure wait_lora_patch is applied to VLLMClient
        wait_lora_patch(max_wait=300, retry_interval=2.0, initial_delay=0.0)

        self.client = VLLMClient(server_url=self.vllm_server_base_url)

        if benchmark_env is None:
            self.benchmark_env = BenchmarkEnv(
                base_model=self.base_model_name,
                vllm_server_base_url=self.vllm_server_base_url,
                output_dir=self.output_dir,
                **kwargs,
            )
        else:
            self.benchmark_env = benchmark_env

    @classmethod
    def from_config(
        cls, config: sws.FinalConfig, tokenizer: PreTrainedTokenizerBase | None = None
    ) -> LoraBenchmarkRunner:
        def _get(*paths, default=None):
            for path in paths:
                try:
                    curr = config
                    for part in path.split("."):
                        curr = getattr(curr, part)
                    if curr is not None:
                        return curr
                except (AttributeError, KeyError):
                    continue
            return default

        base_model = _get("model.model_name", "model_name", "model.model", default="Qwen/Qwen2.5-Math-1.5B")
        vllm_url = _get("model.vllm_server_base_url", "vllm_server_base_url", "grpo.vllm_server_base_url", default="http://localhost:8000")
        checkpoints_dir = _get("checkpoints_dir", "eval.checkpoints_dir", "output_dir", default=None)
        explicit_checkpoints = _get("eval.checkpoints", "checkpoints", default=None)
        eval_base = _get("eval.eval_base_model", "eval_base_model", default=False)
        output_dir = _get("eval.output_dir", "output_dir", default="outputs/bench")
        report_to = _get("eval.report_to", "report_to", "grpo.report_to", default=None)
        project_name = _get("project_name", default="lora-bench")
        exp_name = _get("exp_name", default="eval")
        run_name = _get("run_name", "grpo.run_name", default=f"{project_name}_{exp_name}")
        overwrite = _get("eval.overwrite", "overwrite", default=False)

        max_comp_len = _get("eval.max_completion_length", "max_completion_length", "grpo.max_completion_length", default=3000)
        temp = _get("eval.temperature", "temperature", default=0.6)
        top_p = _get("eval.top_p", "top_p", default=0.95)
        top_k = _get("eval.top_k", "top_k", default=None)
        concurrency = _get("eval.concurrency", "concurrency", default=8)
        log_comp = _get("eval.log_completions", "log_completions", default=True)
        num_print = _get("eval.num_completions_to_print", "num_completions_to_print", default=5)
        max_iters = _get("eval.max_tool_calling_iterations", "max_tool_calling_iterations", default=10)
        chat_kwargs = _get("eval.chat_template_kwargs", "chat_template_kwargs", default={"enable_thinking": True})

        bench_env = BenchmarkEnv(
            base_model=base_model,
            tokenizer=tokenizer,
            vllm_server_base_url=vllm_url,
            max_completion_length=max_comp_len,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            chat_template_kwargs=chat_kwargs.to_dict() if hasattr(chat_kwargs, "to_dict") else chat_kwargs,
            max_tool_calling_iterations=max_iters,
            concurrency=concurrency,
            log_completions=log_comp,
            num_completions_to_print=num_print,
            output_dir=output_dir,
        )

        return cls(
            base_model_name=base_model,
            vllm_server_base_url=vllm_url,
            checkpoints_dir=checkpoints_dir,
            explicit_checkpoints=explicit_checkpoints,
            eval_base_model=eval_base,
            output_dir=output_dir,
            benchmark_env=bench_env,
            report_to=report_to,
            project_name=project_name,
            exp_name=exp_name,
            run_name=run_name,
            overwrite=overwrite,
        )

    def run(
        self,
        dataset: Dataset | Sequence[dict[str, Any]],
        env_factory: Callable | None = None,
        reward_funcs: Any | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Run the full evaluation over all discovered checkpoints."""
        self.benchmark_env.wait_for_vllm()

        checkpoints = discover_checkpoints(
            checkpoints_dir=self.checkpoints_dir,
            explicit_checkpoints=self.explicit_checkpoints,
            eval_base_model=self.eval_base_model,
            base_model_name=self.base_model_name,
        )

        if not checkpoints:
            logger.warning(
                f"No checkpoints found in {self.checkpoints_dir} and eval_base_model=False. "
                f"Falling back to evaluating base model '{self.base_model_name}'."
            )
            checkpoints = [("base_model", None, 0)]

        logger.info(
            f"Discovered {len(checkpoints)} evaluation targets: "
            f"{[c[0] for c in checkpoints]}"
        )

        # Trackio run setup if requested
        trackio_run = None
        if self.report_to == "trackio":
            try:
                import trackio
                trackio_run = trackio.init(
                    project=self.project_name,
                    name=self.run_name,
                    resume="allow",
                )
                logger.info(f"Attached to Trackio run '{self.run_name}' in project '{self.project_name}'")
            except Exception as e:
                logger.warning(f"Could not initialize Trackio: {e}")

        all_metrics: list[dict[str, Any]] = []

        for name, path, step in checkpoints:
            clean_name = re.sub(r"[^\w\-_\.]", "_", name)
            parquet_path = self.output_dir / f"{clean_name}.parquet"

            if parquet_path.exists() and not self.overwrite:
                logger.info(f"Skipping already completed evaluation for '{name}' (found {parquet_path})")
                try:
                    df = pd.read_parquet(parquet_path)
                    total_episodes = len(df)
                    rewards = df["reward"].values if not df.empty else np.array([])
                    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
                    std_reward = float(np.std(rewards)) if len(rewards) > 0 else 0.0
                    sem_reward = float(std_reward / np.sqrt(len(rewards))) if len(rewards) > 0 else 0.0
                    min_reward = float(np.min(rewards)) if len(rewards) > 0 else 0.0
                    max_reward = float(np.max(rewards)) if len(rewards) > 0 else 0.0
                    total_wins = int(df["won"].sum()) if not df.empty else 0
                    win_rate = float(total_wins / total_episodes) if total_episodes > 0 else 0.0
                    mean_turns = float(df["turns"].mean()) if not df.empty else 0.0
                    mean_tokens = float(df["completion_tokens"].mean()) if not df.empty else 0.0
                    mean_duration = float(df["duration_s"].mean()) if not df.empty else 0.0

                    metrics = {
                        "checkpoint_name": name,
                        "step": step,
                        "model_name": self.base_model_name if path is None else f"eval-{name}",
                        "total_episodes": total_episodes,
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "sem_reward": sem_reward,
                        "min_reward": min_reward,
                        "max_reward": max_reward,
                        "wins": total_wins,
                        "win_rate": win_rate,
                        "mean_turns": mean_turns,
                        "mean_completion_tokens": mean_tokens,
                        "mean_duration_s": mean_duration,
                        "parquet_path": str(parquet_path),
                    }
                    all_metrics.append(metrics)

                    if trackio_run is not None:
                        try:
                            import trackio
                            trackio.log(
                                {
                                    "eval/mean_reward": metrics["mean_reward"],
                                    "eval/std_reward": metrics["std_reward"],
                                    "eval/sem_reward": metrics["sem_reward"],
                                    "eval/min_reward": metrics["min_reward"],
                                    "eval/max_reward": metrics["max_reward"],
                                    "eval/win_rate": metrics["win_rate"],
                                    "eval/wins": metrics["wins"],
                                    "eval/mean_turns": metrics["mean_turns"],
                                    "eval/mean_completion_tokens": metrics["mean_completion_tokens"],
                                    "eval/mean_duration_s": metrics["mean_duration_s"],
                                    "eval/step": step,
                                },
                                step=step,
                            )
                        except Exception as e:
                            logger.warning(f"Failed logging cached metrics to Trackio: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed loading cached parquet for {name}: {e}. Re-evaluating.")

            logger.info(f"\n{'='*70}\n[START EVALUATION] Target: {name} (step: {step})\n{'='*70}")
            active_model = self.base_model_name

            if path is not None:
                # Load LoRA adapter into vLLM
                lora_name = f"eval-{name}"
                logger.info(f"Loading LoRA adapter '{lora_name}' from {path} into vLLM...")
                try:
                    self.client.load_lora_adapter(lora_name=lora_name, lora_path=str(path.resolve()))
                    active_model = lora_name
                except Exception as e:
                    logger.error(f"Failed to load LoRA adapter for {name} from {path}: {e}")
                    continue
            else:
                lora_name = None

            try:
                metrics, _ = asyncio.run(
                    self.benchmark_env.eval_checkpoint(
                        dataset=dataset,
                        model_name=active_model,
                        checkpoint_name=name,
                        step=step,
                        env_factory=env_factory,
                        reward_funcs=reward_funcs,
                    )
                )
                all_metrics.append(metrics)

                if trackio_run is not None:
                    try:
                        import trackio
                        trackio.log(
                            {
                                "eval/mean_reward": metrics["mean_reward"],
                                "eval/std_reward": metrics["std_reward"],
                                "eval/sem_reward": metrics["sem_reward"],
                                "eval/win_rate": metrics["win_rate"],
                                "eval/mean_turns": metrics["mean_turns"],
                                "eval/mean_completion_tokens": metrics["mean_completion_tokens"],
                                "eval/mean_duration_s": metrics["mean_duration_s"],
                                "eval/step": step,
                            },
                            step=step,
                        )
                    except Exception as e:
                        logger.warning(f"Failed logging to Trackio: {e}")

            finally:
                if lora_name is not None:
                    logger.info(f"Unloading LoRA adapter '{lora_name}' from vLLM...")
                    try:
                        self.client.unload_lora_adapter(lora_name=lora_name)
                    except Exception as e:
                        logger.warning(f"Failed unloading LoRA adapter {lora_name}: {e}")

        summary_df = pd.DataFrame(all_metrics)

        # Save summary files
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path = self.output_dir / "summary.json"
        summary_parquet_path = self.output_dir / "summary.parquet"
        summary_csv_path = self.output_dir / "summary.csv"

        with open(summary_json_path, "w") as f:
            json.dump({"checkpoints": all_metrics}, f, indent=2)

        if not summary_df.empty:
            summary_df.to_parquet(summary_parquet_path, index=False)
            summary_df.to_csv(summary_csv_path, index=False)

        # Print full benchmark comparison table
        print("\n" + "=" * 80)
        print("                 BENCHMARK CHECKPOINT EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Base Model:      {self.base_model_name}")
        print(f"Checkpoints Dir: {self.checkpoints_dir}")
        print(f"Output Dir:      {self.output_dir}")
        print("-" * 80)
        if not summary_df.empty:
            cols = ["checkpoint_name", "step", "mean_reward", "std_reward", "sem_reward", "win_rate", "total_episodes"]
            display_cols = [c for c in cols if c in summary_df.columns]
            print(summary_df[display_cols].to_string(index=False))
        else:
            print("No checkpoints evaluated successfully.")
        print("=" * 80 + "\n")

        return {"checkpoints": all_metrics}, summary_df


def main(config: sws.FinalConfig):
    env_dict = make_env(config.env_name, config.env.to_dict())
    tokenizer = env_dict.get("processing_class") or env_dict.get("tokenizer")
    runner = LoraBenchmarkRunner.from_config(config, tokenizer=tokenizer)

    dataset = (
        env_dict.get("eval_dataset")
        or env_dict.get("test_dataset")
        or env_dict.get("dataset")
        or env_dict.get("train_dataset")
    )

    reward_funcs = env_dict.get("reward_funcs")
    env_factory = env_dict.get("environment_factory") or env_dict.get("env_factory")

    summary, df = runner.run(
        dataset=dataset,
        env_factory=env_factory,
        reward_funcs=reward_funcs,
    )
    return summary


def main_cli():
    sws_run(main)


if __name__ == "__main__":
    main_cli()
