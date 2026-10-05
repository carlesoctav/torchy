"""Monkey-patch VLLMClient.load_lora_adapter to retry until the adapter is visible on vLLM's mount.

Useful in multi-server Async GRPO setups with remote storage (e.g., HF Storage Buckets),
where writes take a few seconds to propagate from the trainer node to the vLLM node.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def wait_lora_patch(
    max_wait: int = 300,
    retry_interval: float = 2.0,
    initial_delay: float = 3.0,
) -> None:
    """Monkey-patch VLLMClient.load_lora_adapter with gating and retry logic.

    Args:
        max_wait: Maximum time in seconds to wait before giving up (default 300s).
        retry_interval: Interval in seconds between retries (default 2.0s).
        initial_delay: Gating delay in seconds before the first attempt (default 3.0s).
            Giving the remote storage mount ~2-3s to flush writes to the Hub ensures
            vLLM's initial open() succeeds on the first try, preventing negative dentry
            cache stalls.
    """
    from trl.experimental.async_grpo.vllm_client import VLLMClient

    if getattr(VLLMClient, "_wait_lora_patched", False):
        return

    orig_load_lora = VLLMClient.load_lora_adapter

    def retrying_load_lora_adapter(self, lora_name: str, lora_path: str, timeout: int = 1800) -> None:
        start = time.time()
        if initial_delay > 0:
            logger.info(
                f"[wait_lora_patch] Waiting {initial_delay:.1f}s for remote mount flush before loading '{lora_name}'..."
            )
            time.sleep(initial_delay)

        attempt = 0
        while True:
            attempt += 1
            try:
                orig_load_lora(self, lora_name, lora_path, timeout=timeout)
                elapsed = time.time() - start
                if attempt > 1:
                    logger.info(
                        f"Successfully loaded LoRA adapter '{lora_name}' after {attempt} attempts "
                        f"({elapsed:.1f}s total)."
                    )
                else:
                    logger.info(f"Loaded LoRA adapter '{lora_name}' on first attempt ({elapsed:.1f}s total).")
                return
            except Exception as e:
                elapsed = time.time() - start
                err_msg = str(e)
                # If the adapter is already registered from a previous run, unload it and retry immediately
                if "already been loaded" in err_msg:
                    logger.info(f"[wait_lora_patch] Adapter '{lora_name}' already loaded in vLLM. Unloading stale adapter first...")
                    try:
                        self.unload_lora_adapter(lora_name)
                    except Exception:
                        pass
                    continue

                # vLLM returns 404 (LoRAAdapterNotFoundError) when the path is not visible yet on the mount.
                # TRL's vllm_client wraps 404 into RuntimeError("The vLLM server at ... does not expose /v1/load_lora_adapter...")
                if elapsed < max_wait:
                    logger.warning(
                        f"[wait_lora_patch] Failed to load adapter '{lora_name}' at '{lora_path}' "
                        f"(attempt {attempt}, elapsed {elapsed:.1f}s / max {max_wait}s): {e}. "
                        f"Retrying in {retry_interval}s..."
                    )
                    time.sleep(retry_interval)
                    continue
                logger.error(
                    f"[wait_lora_patch] Timed out waiting for adapter '{lora_name}' at '{lora_path}' "
                    f"after {elapsed:.1f}s."
                )
                raise e

    VLLMClient.load_lora_adapter = retrying_load_lora_adapter
    VLLMClient._wait_lora_patched = True
    logger.info("Applied wait_lora_patch to VLLMClient.load_lora_adapter.")

