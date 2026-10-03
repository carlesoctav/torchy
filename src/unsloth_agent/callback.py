"""Bucket sync for run outputs: local-first, never blocks training.

trackio logs to ``TRACKIO_DIR`` (set to ``outputs/{project}/trackio`` in
:meth:`train_grpo.main`, i.e. project level so one dashboard shows all runs).
This callback mirrors the whole project dir — checkpoints, completions, and
the trackio db — to the bucket on every save and once at the end. Every
failure warns; training never dies because a sync failed. No ``bucket_id``
in config disables syncing silently.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from transformers import TrainerCallback


def ensure_bucket(bucket_id: str) -> None:
    """Create the bucket if missing (exist_ok). Called once per run, not per
    sync. Import is lazy: hub/trackio are training-time dependencies."""
    from trackio.bucket_storage import create_bucket_if_not_exists

    create_bucket_if_not_exists(bucket_id)


def project_dir_for(output_dir: str) -> str:
    """The sync root: the project dir (parent of the exp dir), so the
    sibling ``trackio/`` dir is included. Falls back to the exp dir itself
    when the output dir has no parent (e.g. bare ``"trainer_output"``)."""
    resolved = Path(output_dir).resolve()
    if resolved.parent == Path.cwd().resolve():
        return str(resolved)
    return str(resolved.parent)


class BucketSyncCallback(TrainerCallback):
    def __init__(self, config):
        self.bucket_id = config.get("bucket_id")
        self.enabled = bool(self.bucket_id)
        self._local_dir = None
        self._shut_down = False
        if self.enabled:
            try:
                ensure_bucket(self.bucket_id)
            except Exception as exc:
                warnings.warn(
                    f"BucketSyncCallback: cannot ensure bucket "
                    f"{self.bucket_id!r}: {exc}. Syncing disabled."
                )
                self.enabled = False

    @property
    def bucket_url(self) -> str:
        return f"hf://buckets/{self.bucket_id}"

    def _sync(self):
        if not self.enabled or self._local_dir is None:
            return
        from huggingface_hub import HfApi  # lazy: training-time dep

        try:
            HfApi().sync_bucket(
                f"{self._local_dir}",
                self.bucket_url,
                exclude=["completions/*", "*/completions/*"],
            )
        except Exception as exc:
            warnings.warn(f"BucketSyncCallback: sync failed, will retry: {exc}")

    def on_save(self, args, state, control, **kwargs):
        if self._local_dir is None:
            self._local_dir = project_dir_for(args.output_dir)
        self._sync()

    def on_train_end(self, args, state, control, **kwargs):
        if self._local_dir is None:
            self._local_dir = project_dir_for(args.output_dir)
        self.shutdown()

    def shutdown(self):
        """Push a final sync. Idempotent: safe to call from both
        ``on_train_end`` and a ``finally`` in ``main`` (crashes skip the
        callback)."""
        if self._shut_down:
            return
        self._shut_down = True
        self._sync()
