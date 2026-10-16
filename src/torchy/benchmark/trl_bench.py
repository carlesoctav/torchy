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
from transformers.utils.chat_template_utils import get_json_schema
from trl.chat_template_utils import add_response_schema, parse_response
from renderers import ToolCallParseStatus, create_renderer


def _convert_tools_to_schemas(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert a list of Python callables or dicts into OpenAI-compatible tool schemas for renderers."""
    if not tools:
        return None
    schemas = []
    for tool in tools:
        if isinstance(tool, dict):
            schemas.append(tool)
        elif callable(tool):
            try:
                schemas.append(get_json_schema(tool))
            except Exception:
                sig = inspect.signature(tool)
                props = {k: {"type": "string"} for k in sig.parameters.keys()}
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "__name__", str(tool)),
                        "description": inspect.getdoc(tool) or f"Call {getattr(tool, '__name__', str(tool))}.",
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": [
                                k for k, p in sig.parameters.items()
                                if p.default == inspect.Parameter.empty
                            ],
                        },
                    },
                })
        else:
            schemas.append(tool)
    return schemas

# Ensure repo root and src/ are importable
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/torchy/benchmark/trl_bench.py -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import sws
from torchy.env import make_env
from torchy.sws_utils import run as sws_run

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BenchmarkEnv")


class BenchmarkEnv:
    """Benchmark evaluation runner mimicking AsyncGRPOTrainer's multi-turn rollout loop.

    Instead of passing generated tokens to an optimizer for policy training, BenchmarkEnv
    evaluates model inference rollouts against an external vLLM server and environment,
    logs sample completions, and saves all generation traces into a single parquet file.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-1.7B",
        tokenizer: PreTrainedTokenizerBase | None = None,
        vllm_server_base_url: str = "http://localhost:8000",
        max_completion_length: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tool_calling_iterations: int = 10,
        concurrency: int = 8,
        log_completions: bool = True,
        num_completions_to_print: int = 10,
        output_dir: str | Path = "outputs/bench",
        parquet_filename: str = "benchmark_results.parquet",
        **kwargs,
    ):
        self.model_name = model
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
        self.parquet_filename = parquet_filename
        self.max_model_len = 8192
        self.extra_kwargs = kwargs

        if tokenizer is None:
            logger.info(f"Loading tokenizer for {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        has_template = getattr(tokenizer, "response_template", None) is not None
        has_schema = getattr(tokenizer, "response_schema", None) is not None
        if not has_template and not has_schema:
            try:
                tokenizer = add_response_schema(tokenizer)
            except Exception as e:
                logger.warning(f"Could not apply add_response_schema: {e}")
        self.tokenizer = tokenizer

        try:
            self.renderer = create_renderer(
                self.tokenizer,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            logger.info(
                f"Initialized renderer {type(self.renderer).__name__} "
                f"(is_prefix_stable={self.renderer.is_prefix_stable})"
            )
        except Exception as e:
            logger.warning(
                f"Could not initialize custom renderer with chat_template_kwargs {self.chat_template_kwargs}: {e}. "
                "Falling back to default renderer."
            )
            self.renderer = create_renderer(self.tokenizer)

    def _parse_completion(
        self,
        turn_ids: list[int],
        prompt_ids: list[int],
        turn_text: str,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Parse completion tokens into an assistant message dict using the renderer."""
        reasoning = ""
        content_text = ""
        parsed_calls = []

        try:
            parsed = self.renderer.parse_response(
                turn_ids,
                prompt_ids=prompt_ids,
                tools=tool_schemas,
            )
            reasoning = parsed.reasoning_content or ""
            content_text = parsed.content or ""
            for i, tc in enumerate(parsed.tool_calls):
                if tc.status == ToolCallParseStatus.OK and tc.name:
                    parsed_calls.append({
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else (tc.arguments or "{}"),
                        },
                    })
        except Exception as e:
            logger.warning(f"renderer.parse_response failed: {e}. Falling back to trl parse_response.")
            assistant_msg = parse_response(self.tokenizer, turn_ids, prefix=prompt_ids)
            reasoning = assistant_msg.get("reasoning_content", "")
            content_text = assistant_msg.get("content", "")
            parsed_calls = assistant_msg.get("tool_calls", [])

        # Fallback regex if tool call was in text but parser missed it
        if not parsed_calls and "<tool_call>" in turn_text:
            m = re.search(
                r"<tool_call>\s*(.*?)\s*</tool_call>", turn_text, re.DOTALL
            )
            if m:
                try:
                    parsed_tc = json.loads(m.group(1))
                    if "name" in parsed_tc:
                        parsed_calls = [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": parsed_tc["name"],
                                    "arguments": json.dumps(parsed_tc.get("arguments", {}))
                                    if isinstance(parsed_tc.get("arguments"), dict)
                                    else (parsed_tc.get("arguments") or "{}"),
                                },
                            }
                        ]
                except Exception:
                    pass

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
            "reasoning_content": reasoning,
        }
        if parsed_calls:
            msg["tool_calls"] = parsed_calls
        return msg

    @classmethod
    def from_config(cls, config: sws.FinalConfig) -> BenchmarkEnv:
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

        model_name = _get("model_name", "model.model_name", "model.model", default="Qwen/Qwen3-1.7B")
        vllm_url = _get("vllm_server_base_url", "model.vllm_server_base_url", "grpo.vllm_server_base_url", default="http://localhost:8000")
        max_comp_len = _get("eval.max_completion_length", "max_completion_length", "grpo.max_completion_length", default=4096)
        temp = _get("eval.temperature", "temperature", default=0.6)
        top_p = _get("eval.top_p", "top_p", default=0.95)
        top_k = _get("eval.top_k", "top_k", default=None)
        presence_penalty = _get("eval.presence_penalty", "presence_penalty", default=None)
        repetition_penalty = _get("eval.repetition_penalty", "repetition_penalty", default=None)
        concurrency = _get("eval.concurrency", "concurrency", "grpo.per_device_train_batch_size", default=8)
        log_comp = _get("eval.log_completions", "log_completions", "grpo.log_completions", default=True)
        num_print = _get("eval.num_completions_to_print", "num_completions_to_print", "grpo.num_completions_to_print", default=10)
        max_iters = _get("eval.max_tool_calling_iterations", "max_tool_calling_iterations", default=10)
        chat_kwargs = _get("eval.chat_template_kwargs", "chat_template_kwargs", "grpo.chat_template_kwargs", default={"enable_thinking": True})
        output_dir = _get("output_dir", default="outputs/bench")
        parquet_filename = _get("parquet_filename", default="wordle_benchmark.parquet")

        return cls(
            model=model_name,
            vllm_server_base_url=vllm_url,
            max_completion_length=max_comp_len,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            chat_template_kwargs=chat_kwargs.to_dict() if hasattr(chat_kwargs, "to_dict") else chat_kwargs,
            max_tool_calling_iterations=max_iters,
            concurrency=concurrency,
            log_completions=log_comp,
            num_completions_to_print=num_print,
            output_dir=output_dir,
            parquet_filename=parquet_filename,
        )

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
    ) -> tuple[list[int] | None, str]:
        """Request a single turn completion from vLLM, clamping tokens to context limit."""
        prompt_len = len(prompt_ids)
        available_tokens = self.max_model_len - prompt_len
        if available_tokens <= 16:
            logger.warning(
                f"Prompt length {prompt_len} reached context limit {self.max_model_len}; stopping turn generation."
            )
            return None, ""

        max_tokens = min(self.max_completion_length, available_tokens)
        payload: dict[str, Any] = {
            "model": self.model_name,
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
                            turn_ids = self.tokenizer.encode(
                                turn_text, add_special_tokens=False
                            )
                        return turn_ids, turn_text
                    else:
                        err_text = await resp.text()
                        logger.warning(f"vLLM status {resp.status}: {err_text}")
            except Exception as e:
                logger.warning(f"vLLM request error on attempt {attempt + 1}: {e}")
            await asyncio.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Failed to generate completion from vLLM after {max_attempts} attempts."
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
                            # Positional fallback if parameter name differs
                            first_val = (
                                list(arguments.values())[0] if arguments else ""
                            )
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
            if (
                "invalid move" in result_str.lower()
                or "wrong format" in result_str.lower()
            ):
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
        outcome = "WON" if record["won"] else "LOST"
        print("\n" + "=" * 80)
        print(
            f"[Episode {idx + 1}] Seed: {record.get('seed')} | Outcome: {outcome} | "
            f"Reward: {record['reward']:.2f} | Turns: {record['turns']} | Time: {record['duration_s']}s"
        )
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
        print(f"Result: {outcome} (Reward: {record['reward']})")
        print("=" * 80 + "\n")

    async def _eval_one_episode(
        self,
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        idx: int,
        row: dict[str, Any],
        env_factory: Callable,
        env_pool: list[Any],
    ) -> dict[str, Any]:
        async with sem:
            t0 = time.time()
            env = env_pool.pop() if env_pool else env_factory()

            reset_kwargs = {
                k: v for k, v in row.items() if k not in ("prompt", "environment")
            }
            try:
                observation = env.reset(**reset_kwargs)
            except Exception as e:
                logger.error(f"[Episode {idx}] env.reset failed: {e}")
                observation = None

            if "prompt" in row:
                prompt_messages = copy.deepcopy(row["prompt"])
            else:
                prompt_messages = [{"role": "user", "content": ""}]

            if observation is not None:
                last_msg = copy.deepcopy(prompt_messages[-1])
                content = last_msg["content"]
                if isinstance(observation, list) and isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(observation, str) and isinstance(content, list):
                    observation = [{"type": "text", "text": observation}]
                last_msg["content"] = content + observation
                prompt_messages = prompt_messages[:-1] + [last_msg]

            messages = copy.deepcopy(prompt_messages)

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
            tool_schemas = _convert_tools_to_schemas(env_methods)

            tool_calls_count = 0
            tool_failures_count = 0
            invalid_moves_count = 0
            turns = 0
            trace_log = []

            # Initial turn prompt tokens using renderer
            prompt_ids = self.renderer.render_ids(
                messages,
                tools=tool_schemas,
                add_generation_prompt=True,
            )
            all_trajectory_tokens = list(prompt_ids)

            while True:
                turn_ids, turn_text = await self._generate_one_turn(
                    session, prompt_ids
                )
                if turn_ids is None:
                    trace_log.append({
                        "turn": turns + 1,
                        "reasoning": "",
                        "content": "Context limit reached.",
                        "stop_reason": "context_limit",
                    })
                    break

                assistant_message = self._parse_completion(
                    turn_ids, prompt_ids, turn_text, tool_schemas=tool_schemas
                )

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
                    all_trajectory_tokens.extend(turn_ids)
                    trace_log.append(turn_trace)
                    break

                if turns >= self.max_tool_calling_iterations:
                    turn_trace["stop_reason"] = "max_iterations"
                    all_trajectory_tokens.extend(turn_ids)
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

                if getattr(env, "done", False):
                    all_trajectory_tokens.extend(turn_ids)
                    break

                # Bridge to next turn without re-tokenizing previous prompt/completions (100% prefix cache preservation)
                bridge = self.renderer.bridge_to_next_turn(
                    previous_prompt_ids=prompt_ids,
                    previous_completion_ids=turn_ids,
                    new_messages=tool_msgs,
                    tools=tool_schemas,
                )
                if bridge is not None:
                    prompt_ids = bridge.token_ids
                else:
                    logger.debug("bridge_to_next_turn returned None, falling back to render_ids")
                    prompt_ids = self.renderer.render_ids(
                        messages,
                        tools=tool_schemas,
                        add_generation_prompt=True,
                    )
                all_trajectory_tokens = list(prompt_ids)

            if hasattr(env, "get_reward"):
                reward = env.get_reward()
                if inspect.iscoroutine(reward):
                    reward = await reward
            elif hasattr(env, "reward"):
                reward = env.reward
            else:
                reward = 0.0

            reward = float(reward)
            won = (reward == 1.0) or any(
                isinstance(m.get("content"), str) and "Congratulations" in m["content"]
                for m in messages
            )

            env_pool.append(env)
            duration_s = time.time() - t0

            secret_word = None
            for m in reversed(messages):
                c = m.get("content", "")
                if isinstance(c, str):
                    m_word = re.search(
                        r"secret word was:? \*\*?([a-zA-Z]+)\*\*?", c, re.IGNORECASE
                    )
                    if m_word:
                        secret_word = m_word.group(1).upper()
                        break

            try:
                rendered_trace = self.tokenizer.decode(all_trajectory_tokens)
            except Exception:
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
                "seed": row.get("seed"),
                "won": bool(won),
                "reward": reward,
                "turns": turns,
                "tool_calls_count": tool_calls_count,
                "tool_failures_count": tool_failures_count,
                "invalid_moves_count": invalid_moves_count,
                "duration_s": round(duration_s, 2),
                "secret_word": secret_word,
                "model_name": self.model_name,
                "prompt_json": json.dumps(prompt_messages),
                "messages_json": json.dumps(messages),
                "trace_json": json.dumps(trace_log),
                "rendered_trace": rendered_trace,
            }

            if self.log_completions and (idx < self.num_completions_to_print):
                self._print_episode_trace(
                    idx, episode_record, prompt_messages, trace_log
                )
            else:
                logger.info(
                    f"Episode {idx + 1} finished | Outcome: {'WON' if won else 'LOST'} | "
                    f"Reward: {reward:.2f} | Turns: {turns} | Time: {duration_s:.1f}s"
                )

            return episode_record

    async def _run_eval(
        self,
        env_factory: Callable,
        datasets: Dataset | Sequence[dict[str, Any]],
        **kwargs,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        self.wait_for_vllm()

        # Convert dataset to list of rows
        if hasattr(datasets, "to_list"):
            rows = datasets.to_list()
        elif hasattr(datasets, "__iter__"):
            rows = [datasets[i] for i in range(len(datasets))]
        else:
            rows = list(datasets)

        total_episodes = len(rows)
        logger.info(
            f"Starting benchmark evaluation on {total_episodes} cases "
            f"(concurrency={self.concurrency}, max_tool_iterations={self.max_tool_calling_iterations})"
        )

        sem = asyncio.Semaphore(self.concurrency)
        env_pool = [env_factory() for _ in range(self.concurrency)]

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.concurrency * 2)
        ) as session:
            tasks = [
                asyncio.create_task(
                    self._eval_one_episode(
                        sem, session, i, row, env_factory, env_pool
                    )
                )
                for i, row in enumerate(rows)
            ]
            records = await asyncio.gather(*tasks)

        df = pd.DataFrame(records)

        # Compute summary metrics
        total_wins = int(df["won"].sum())
        win_rate = float(total_wins / total_episodes) if total_episodes > 0 else 0.0
        mean_reward = float(df["reward"].mean())
        mean_turns = float(df["turns"].mean())
        mean_duration = float(df["duration_s"].mean())
        total_tool_calls = int(df["tool_calls_count"].sum())
        total_tool_failures = int(df["tool_failures_count"].sum())
        total_invalid_moves = int(df["invalid_moves_count"].sum())

        metrics = {
            "total_episodes": total_episodes,
            "wins": total_wins,
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "mean_turns": mean_turns,
            "mean_duration_s": mean_duration,
            "total_tool_calls": total_tool_calls,
            "total_tool_failures": total_tool_failures,
            "total_invalid_moves": total_invalid_moves,
        }

        # Save single parquet file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self.output_dir / self.parquet_filename
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved all generation records to parquet: {parquet_path}")

        summary_json_path = self.output_dir / "summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Print summary report
        print("\n" + "=" * 60)
        print("          BENCHMARK EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model:                 {self.model_name}")
        print(f"Total Episodes:        {total_episodes}")
        print(f"Wins:                  {total_wins} / {total_episodes}")
        print(f"Win Rate:              {win_rate * 100:.2f}%")
        print(f"Mean Reward:           {mean_reward:.4f}")
        print(f"Mean Turns / Guesses:  {mean_turns:.2f}")
        print(f"Mean Duration / Game:  {mean_duration:.2f}s")
        print(f"Total Tool Calls:      {total_tool_calls}")
        print(f"Total Tool Failures:   {total_tool_failures}")
        print(f"Total Invalid Moves:   {total_invalid_moves}")
        print(f"Parquet Output:        {parquet_path}")
        print("=" * 60 + "\n")

        return metrics, df

    def eval(
        self,
        env_factory: Callable | None = None,
        datasets: Dataset | Sequence[dict[str, Any]] | None = None,
        environment_factory: Callable | None = None,
        dataset: Dataset | Sequence[dict[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        factory = env_factory or environment_factory
        data = datasets if datasets is not None else dataset
        if factory is None:
            raise ValueError(
                "env_factory (or environment_factory) must be provided to eval()"
            )
        if data is None:
            raise ValueError("datasets (or dataset) must be provided to eval()")

        return asyncio.run(self._run_eval(factory, data, **kwargs))


def main(config: sws.FinalConfig):
    env_dict = make_env(config.env_name, config.env.to_dict())
    benchmark_env = BenchmarkEnv.from_config(config)

    metrics, df = benchmark_env.eval(
        env_factory=env_dict["environment_factory"],
        datasets=env_dict["train_dataset"],
    )
    return metrics


if __name__ == "__main__":
    sws_run(main)
