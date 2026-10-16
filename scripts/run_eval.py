#!/usr/bin/env python3
"""Launcher script for AsyncGRPO + LoRA reproducing https://huggingface.co/blog/asyncgrpo-lora-hfjobs.
Mimics prime-rl/src/prime_rl/entrypoints/rl.py using ColabSession, ColabProcess, and ProcessGroup.

Orchestrates 2 Colab VMs:
  1. Inference node (vLLM server) with -L local port forwarding.
  2. Trainer node (TRL AsyncGRPO) with -R remote port forwarding and HF bucket sync.

Can be run as:
  python scripts/run_combine_async_grpo.py --config configs/combine_repro_async_grpo_lora.py [--reuse] [--stop-vm] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import sws
from torchy.infra import ColabProcess, ColabSession, ProcessGroup
from torchy.sws_utils import load_config_builder, run as sws_run, _split_config_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_combine_async_grpo")


def get_config() -> sws.Config:
    """Default config factory when no --config flag is provided."""
    return load_config_builder(str(root_dir / "configs/combine_repro_async_grpo_lora.py"))


def wait_for_vllm_ready(port: int = 8000, timeout: int = 300) -> bool:
    """Poll vLLM health check endpoint until it responds with 200 OK."""
    health_url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    logger.info(f"Checking vLLM health on {health_url}...")
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("vLLM server is healthy and ready!")
                    return True
        except Exception:
            pass
        time.sleep(3)
    logger.warning("vLLM health check timed out. Proceeding anyway...")
    return False


def build_vllm_command(cfg: sws.FinalConfig) -> list[str]:
    cmd = [
        "uv", "run", "vllm", "serve", cfg.inference.model,
        "--host", str(cfg.inference.host),
        "--port", str(cfg.inference.port),
        "--max-model-len", str(cfg.inference.max_model_len),
        "--gpu-memory-utilization", str(cfg.inference.gpu_memory_utilization),
        "--logprobs-mode", str(cfg.inference.logprobs_mode),
        "--generation-config", str(cfg.inference.generation_config),
    ]
    if getattr(cfg.inference, "enable_lora", False):
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(cfg.inference.max_lora_rank),
            "--max-loras", str(cfg.inference.max_loras),
        ])
    return cmd


def build_trainer_command(cfg: sws.FinalConfig, config_rel_path: str) -> list[str]:
    return [
        "uv", "run", "python", "-m", "torchy.train.train_async_grpo",
        "--config", config_rel_path,
    ]


def main(cfg: sws.FinalConfig, unused_argv: list[str] | None = None):
    # Parse CLI flags that override or augment the sws config
    parser = argparse.ArgumentParser(description="Run 2-VM AsyncGRPO + LoRA Reproduction")
    parser.add_argument("--stop-vm", action="store_true", help="Stop Colab VMs upon completion or interrupt")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing Colab sessions without recreating them")
    parser.add_argument("--gpu", default=None, help="Override GPU accelerator (e.g. t4, l4, a100)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting VMs")
    extra_args, _ = parser.parse_known_args(unused_argv or [])

    # Extract the original config path from sys.argv so we can pass it to the remote trainer VM
    config_paths, _ = _split_config_args(sys.argv[1:], config_flag="--config")
    if config_paths:
        raw_config_path = config_paths[0]
    else:
        raw_config_path = "configs/combine_repro_async_grpo_lora.py"

    config_full_path = (root_dir / raw_config_path).resolve() if not Path(raw_config_path).is_absolute() else Path(raw_config_path)
    try:
        config_rel_path = str(config_full_path.relative_to(root_dir))
    except ValueError:
        config_rel_path = str(raw_config_path)

    torchy_dir = getattr(cfg.infra, "torchy_dir", "/content/torchy")

    # ColabSession kwargs from config.infra.inference and config.infra.train
    inf_session_kwargs = cfg.infra.inference.to_dict()
    if extra_args.reuse:
        inf_session_kwargs["reuse"] = True
    if extra_args.stop_vm:
        inf_session_kwargs["stop_vm"] = True
    if extra_args.gpu:
        inf_session_kwargs["gpu"] = extra_args.gpu

    train_session_kwargs = cfg.infra.train.to_dict()
    if extra_args.reuse:
        train_session_kwargs["reuse"] = True
    if extra_args.stop_vm:
        train_session_kwargs["stop_vm"] = True
    if extra_args.gpu:
        train_session_kwargs["gpu"] = extra_args.gpu

    logger.info("=== Configuration Loaded via sws ===")
    logger.info(f"Config path: {config_rel_path}")
    logger.info(f"Experiment: {cfg.exp_name} | Project: {cfg.project_name}")
    logger.info(f"Inference Session config: {inf_session_kwargs}")
    logger.info(f"Trainer Session config: {train_session_kwargs}")
    logger.info(f"vLLM model: {cfg.inference.model} | Port: {cfg.inference.port}")

    vllm_cmd = build_vllm_command(cfg)
    trainer_cmd = build_trainer_command(cfg, config_rel_path)

    if extra_args.dry_run:
        logger.info("[Dry Run] vLLM command: " + " ".join(vllm_cmd))
        logger.info("[Dry Run] Trainer command: " + " ".join(trainer_cmd))
        return

    log_dir = root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "vllm.log").unlink(missing_ok=True)
    (log_dir / "trainer.log").unlink(missing_ok=True)

    # 1. Initialize Colab Sessions
    logger.info("Initializing Colab Sessions...")
    vllm_session = ColabSession(**inf_session_kwargs)
    trainer_session = ColabSession(**train_session_kwargs)

    # 2. Launch processes and monitor via ProcessGroup
    with ProcessGroup() as pg:
        logger.info(f"Starting vLLM server on '{vllm_session.session}'...")
        vllm_env = cfg.inference.to_dict().get("env_vars", {})
        vllm_proc = vllm_session.Popen(
            cmd=vllm_cmd,
            log_path=log_dir / "vllm.log",
            dir={str(root_dir): torchy_dir},
            env=vllm_env,
        )
        pg.add("vllm", vllm_proc, session=vllm_session)

        # Wait for vLLM to initialize before starting trainer
        vllm_port = int(cfg.inference.port)
        wait_for_vllm_ready(port=vllm_port, timeout=300)

        logger.info(f"Starting trainer on '{trainer_session.session}'...")
        trainer_proc = trainer_session.Popen(
            cmd=trainer_cmd,
            log_path=log_dir / "trainer.log",
            dir={str(root_dir): torchy_dir},
        )
        pg.add("trainer", trainer_proc, session=trainer_session)

        logger.info("Both nodes running! Monitoring processes...")
        # Mirroring prime-rl: trainer completion is the stop condition
        pg.wait("trainer")
        logger.info("Trainer finished successfully! Cleaning up...")


def main_cli():
    sws_run(main, forward_extras=True)


if __name__ == "__main__":
    main_cli()

