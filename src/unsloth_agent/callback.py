from huggingface_hub import HfApi
from transformers import TrainerCallback


class BucketSyncCallback(TrainerCallback):
    def __init__(self, config):
        self.bucket_url = (
            f"hf://buckets/{config.bucket_id}/{config.exp_name}" # e.g. "hf://buckets/carlesoctav/test/grpo-run"
        )

    def on_save(self, args, state, control, **kwargs):
        HfApi().sync_bucket(
            f"{args.output_dir}",
            f"{self.bucket_url}",
            exclude= ["completions/*"]
        )
