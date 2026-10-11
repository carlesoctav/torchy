"""Monkey-patch VLLMClient.load_lora_adapter to eliminate race conditions
when syncing LoRA weights across distributed nodes (e.g., HF Storage Buckets / FUSE mounts).

When the trainer publishes a new adapter to a remote filesystem, the inference worker's
mount may take a few seconds to see the new files. During this window, vLLM returns
HTTP 404 with {"error": {"message": "... No adapter found for <path> ..."}}.

Standard TRL treats all 404 responses as indicating that `/v1/load_lora_adapter` is not enabled,
raising a fatal RuntimeError.

This patch directly handles the HTTP communication with vLLM's `/v1/load_lora_adapter`:
1. Distinguishes genuine missing-endpoint 404 from "No adapter found" (storage propagation delay).
2. Retries with backoff until the remote mount catches up and vLLM successfully loads the adapter.
3. Automatically unloads stale adapter versions if "already been loaded" is returned.
"""

from __future__ import annotations

import logging
import time
import requests

logger = logging.getLogger(__name__)


def wait_lora_patch(
    max_wait: int = 300,
    retry_interval: float = 2.0,
    initial_delay: float = 1.0,
) -> None:
    """Monkey-patch VLLMClient.load_lora_adapter with robust retry and error handling.

    Args:
        max_wait: Maximum seconds to wait for remote storage propagation (default 300s).
        retry_interval: Seconds between retries (default 2.0s).
        initial_delay: Initial sleep before the first attempt to give FUSE/Hub time to flush (default 1.0s).
    """
    from trl.experimental.async_grpo.vllm_client import VLLMClient

    if getattr(VLLMClient, "_wait_lora_patched", False):
        return

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
            url = f"{self.server_url}/v1/load_lora_adapter"
            payload = {"lora_name": lora_name, "lora_path": lora_path, "load_inplace": False}

            try:
                response = requests.post(url, json=payload, timeout=timeout)
            except Exception as e:
                elapsed = time.time() - start
                if elapsed < max_wait:
                    logger.warning(
                        f"[wait_lora_patch] Connection error calling {url} (attempt {attempt}, elapsed {elapsed:.1f}s): {e}. "
                        f"Retrying in {retry_interval}s..."
                    )
                    time.sleep(retry_interval)
                    continue
                raise

            elapsed = time.time() - start

            if response.status_code == 200:
                if attempt > 1:
                    logger.info(
                        f"[wait_lora_patch] Successfully loaded LoRA adapter '{lora_name}' after {attempt} attempts ({elapsed:.1f}s total)."
                    )
                else:
                    logger.info(f"[wait_lora_patch] Loaded LoRA adapter '{lora_name}' on first attempt ({elapsed:.1f}s total).")
                return

            resp_text = response.text

            # Case 1: Adapter already loaded from a previous run or race
            if response.status_code == 400 and "already been loaded" in resp_text:
                logger.info(f"[wait_lora_patch] Adapter '{lora_name}' already loaded in vLLM. Unloading stale adapter first...")
                try:
                    self.unload_lora_adapter(lora_name)
                except Exception as unload_err:
                    logger.warning(f"[wait_lora_patch] Failed to unload stale adapter '{lora_name}': {unload_err}")
                continue

            # Case 2: Remote mount delay - vLLM returns 404 with "No adapter found"
            if response.status_code in (404, 500) and "No adapter found" in resp_text:
                if elapsed < max_wait:
                    logger.info(
                        f"[wait_lora_patch] Received HTTP {response.status_code} ('No adapter found'). "
                        f"Adapter '{lora_name}' at '{lora_path}' not visible on vLLM mount yet "
                        f"(attempt {attempt}, elapsed {elapsed:.1f}s / max {max_wait}s). Retrying in {retry_interval}s..."
                    )
                    time.sleep(retry_interval)
                    continue
                else:
                    logger.error(
                        f"[wait_lora_patch] Timed out waiting for adapter '{lora_name}' at '{lora_path}' after {elapsed:.1f}s."
                    )
                    raise TimeoutError(
                        f"Timed out after {elapsed:.1f}s waiting for adapter '{lora_name}' at '{lora_path}' to appear on vLLM mount: {resp_text}"
                    )

            # Case 3: Endpoint genuinely missing (404 without "No adapter found")
            if response.status_code == 404:
                raise RuntimeError(
                    f"The vLLM server at {self.server_url} does not expose `/v1/load_lora_adapter`. "
                    f"Restart it with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment. Response: {resp_text}"
                )

            # Case 4: Transient 5xx server error
            if elapsed < max_wait and response.status_code >= 500:
                logger.warning(
                    f"[wait_lora_patch] Server error {response.status_code}: {resp_text}. Retrying in {retry_interval}s..."
                )
                time.sleep(retry_interval)
                continue

            raise RuntimeError(f"Failed to load LoRA adapter '{lora_name}': HTTP {response.status_code} - {resp_text}")

    VLLMClient.load_lora_adapter = retrying_load_lora_adapter
    VLLMClient._wait_lora_patched = True
    logger.info("Applied wait_lora_patch to VLLMClient.load_lora_adapter.")

