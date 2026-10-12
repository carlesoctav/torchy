# Multi-Server Async GRPO with LoRA Reproduction

Reproducing the Hugging Face reference recipe for multi-server Asynchronous GRPO with LoRA training across separate generation and training nodes using a shared Hugging Face Storage Bucket.

- **Reference Blog**: [Hugging Face Async GRPO with LoRA & HF Jobs](https://huggingface.co/blog/asyncgrpo-lora-hfjobs)
- **Reference Code**: [AmineDiro/hfjobs-lora-buckets](https://github.com/AmineDiro/hfjobs-lora-buckets)
- **Training Repository**: [carlesoctav/torchy](https://github.com/carlesoctav/torchy) (branch `hf-repro`)
- **Trackio Dashboard**: [https://trackio.intelligences.systems](https://trackio.intelligences.systems)
  - **Project**: `async-grpo-lora-hf-repro`
  - **Run**: `async-grpo-lora-hf-repro_qwen2.5-math-1.5b-r1-b128`

---

## 1. Architecture & Infrastructure

The training pipeline runs across two decoupled Google Colab runtime nodes (NVIDIA L4 24GB GPUs) communicating over SSH and sharing weights via this Hugging Face Storage Bucket (`carlesoctav/multi-serverasync-grpo` mounted at `/content/lora`):

```
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│        Node 1: Generator        │                 │         Node 2: Trainer         │
│          (vLLM Server)          │                 │          (Async GRPOTrainer)    │
│                                 │                 │                                 │
│  • Qwen/Qwen2.5-Math-1.5B       │  HTTP / SSH     │  • Micro-batch: 8 samples       │
│  • Dynamic LoRA Hot-Swap        │◄───────────────►│  • Grad Accum: 16 (Eff Batch 128│
│  • Max model len: 4096          │  port 8000      │  • FlashAttention-3 (kernels)   │
└────────────────┬────────────────┘                 └────────────────┬────────────────┘
                 │                                                   │
                 │   Shared HF Storage Bucket (/content/lora)        │
                 └───────────────────────►◄──────────────────────────┘
                              Adapter Weight Syncing
                                (every 4-5 steps)
```

---

## 2. Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-Math-1.5B` | Reference causal language model |
| **Dataset** | `sail/Sanity-Test-R1D-1.5B` | Train split of math reasoning prompts |
| **LoRA Rank ($r$)** | `1` | Minimal rank matching HF reference |
| **LoRA Alpha ($\alpha$)** | `2` | Scaling factor ($\alpha / r = 2.0$) |
| **LoRA Target Modules** | `all-linear` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Effective Batch Size** | **`128` completions / step** | 16 prompts $\times$ 8 generations per prompt |
| **Micro-batch Size** | `8` | 1 prompt (8 completions) forwarded immediately |
| **Gradient Accumulation** | `16` | $8 \text{ completions} \times 16 = \mathbf{128\text{ total completions per step}}$ |
| **Token Budget** | `0` | Forces TRL `FixedCountBatcher` (prevents truncation to 4096 tokens) |
| **Total Optimizer Steps** | `500` | Full training run (64,000 completions trained) |
| **Learning Rate** | `4e-5` | Constant learning rate schedule |
| **Optimizer** | `AdamW` | `betas=(0.9, 0.95)`, `weight_decay=0.0` |
| **Precision** | `bfloat16` | Full bf16 with gradient checkpointing |
| **Attention Kernel** | `flash-attn3` | Hugging Face `kernels-community/flash-attn3` |
| **Generations per Prompt ($G$)** | `8` | Parallel rollouts for advantage computation |
| **Sampling Temperature** | `1.0` | Temperature for exploratory rollout generation |
| **Top-p** | `1.0` | Top-p nucleus sampling |
| **Max Completion Length** | `3000` | Max tokens generated per completion |
| **Max Model Length** | `4096` | Context window size for vLLM |
| **Max Staleness** | `4` | Maximum policy lag permitted for rollouts |
| **Max In-Flight Tasks** | `128` | Concurrency limit on async generation pipeline |
| **Weight Sync Interval** | `4` steps | Adapter exported to bucket and hot-swapped in vLLM |
| **Checkpoint Interval** | Every `50` steps | Full optimizer state and adapter saved |
| **Reward Function** | `qwen_math_reward` | Exact match on boxed/answer math solutions |

---

## 3. Reproduction Commands

To reproduce this run using the [`carlesoctav/torchy`](https://github.com/carlesoctav/torchy) repository:

### Step 1: Launch vLLM Rollout Server (Node 1)
```bash
git clone https://github.com/carlesoctav/torchy.git -b hf-repro
cd torchy

# Start vLLM with LoRA hot-swapping enabled and bucket mounted at /content/lora
./scripts/run_vllm.sh
```

Or manually:
```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 VLLM_SERVER_DEV_MODE=1 uv run vllm serve Qwen/Qwen2.5-Math-1.5B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --logprobs-mode processed_logprobs \
    --generation-config vllm \
    --enable-lora \
    --max-lora-rank 1 \
    --max-loras 6
```

### Step 2: Launch Trainer (Node 2)
```bash
git clone https://github.com/carlesoctav/torchy.git -b hf-repro
cd torchy

# Run Async GRPO with LoRA training
uv run python -m torchy.train.train_async_grpo --config configs/asyncgrpo-lora-hf-jobs-repro.py
```

---

## 4. Step-by-Step Reward Progression

Full 500-step training trajectory aggregated across 50-step intervals:

| Step Range | Min Reward | Mean Reward | Max Peak | Checkpoint Saved |
| :--- | :---: | :---: | :---: | :--- |
| **Steps 001 – 050** | 0.0703 | **0.1520** | 0.3047 | `checkpoint-50` |
| **Steps 051 – 100** | 0.0703 | **0.2167** | 0.4844 | `checkpoint-100` |
| **Steps 101 – 150** | 0.1250 | **0.2970** | 0.5625 | `checkpoint-150` |
| **Steps 151 – 200** | 0.1484 | **0.3517** | 0.6250 | `checkpoint-200` |
| **Steps 201 – 250** | 0.1406 | **0.3866** | 0.5703 | `checkpoint-250` |
| **Steps 251 – 300** | 0.1875 | **0.4047** | 0.6719 | `checkpoint-300` |
| **Steps 301 – 350** | 0.1641 | **0.4111** | 0.6094 | `checkpoint-350` |
| **Steps 351 – 400** | 0.1797 | **0.4238** | 0.5938 | `checkpoint-400` |
| **Steps 401 – 450** | 0.2109 | **0.4311** | 0.6406 | `checkpoint-450` |
| **Steps 451 – 500** | 0.1641 | **0.4509** | 0.6406 | `checkpoint-500` & `final-adapter/` |

- **Initial Baseline**: ~0.152
- **Hugging Face Blog Target**: ~0.438
- **Final Achieved Mean**: **`0.4509`**
- **All-Time Peak Single Step**: **`0.6719`** (at step 278)

---

## 5. Artifact Directory Structure in this Bucket

```text
hf://buckets/carlesoctav/multi-serverasync-grpo/async-grpo-lora-hf-repro/qwen2.5-math-1.5b-r1-b128/
├── checkpoint-50/
├── checkpoint-100/
├── checkpoint-150/
├── checkpoint-200/
├── checkpoint-250/
├── checkpoint-300/
├── checkpoint-350/
├── checkpoint-400/
├── checkpoint-450/
├── checkpoint-500/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── final-adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (4.67 MB)
│   └── README.md
├── config.json
└── README.md
```

---

## 6. Metrics & Logging

Live training run logs and GPU hardware telemetries streamed to:
- **Trackio**: [https://trackio.intelligences.systems](https://trackio.intelligences.systems)
- **Project**: `async-grpo-lora-hf-repro`
- **Run**: `async-grpo-lora-hf-repro_qwen2.5-math-1.5b-r1-b128`
- **Total Training Duration**: 10h 32m 11s (avg 72.8s / optimizer step)
- **Total Compute Units Used**: ~38.5 Colab Compute Units (NVIDIA L4 Standard shape @ 3.08 CU/hr combined)
