# CosyVoice3 Benchmark

Benchmarks for CosyVoice3 text-to-speech models with voice cloning, comparing vLLM-Omni streaming serving against HuggingFace Transformers offline inference.

## Prerequisites

```bash
pip install matplotlib aiohttp soundfile numpy tqdm
pip install cosyvoice  # for HF baseline
```

## Quick Start

Run the full benchmark (vllm-omni + HF baseline) with a single command:

```bash
cd benchmarks/cosyvoice3
bash run_benchmark.sh
```

Results (JSON + PNG plots) are saved to `results/`.

### Common options

```bash
# Only vllm-omni (skip HF baseline)
bash run_benchmark.sh --async-only

# Only HF baseline
bash run_benchmark.sh --hf-only

# Use a different model
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B bash run_benchmark.sh --async-only

# Use batch size 16 for higher throughput
BATCH_SIZE=16 bash run_benchmark.sh --async-only

# Custom GPU, prompt count, concurrency levels
GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
```

## Manual Steps

### 1) Start the vLLM-Omni server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" \
    --omni --host 127.0.0.1 --port 8000 \
    --deploy-config vllm_omni/deploy/cosyvoice3.yaml \
    --stage-overrides '{"0":{"max_num_seqs":1,"gpu_memory_utilization":0.4,"max_num_batched_tokens":512},"1":{"max_num_seqs":1,"gpu_memory_utilization":0.2,"max_num_batched_tokens":8192}}' \
    --trust-remote-code
```

### 2) Run online serving benchmark

```bash
python benchmarks/cosyvoice3/vllm_omni/bench_tts_serve.py \
    --port 8000 \
    --num-prompts 50 \
    --max-concurrency 1 4 10 \
    --config-name "async_chunk" \
    --result-dir results/
```

### 3) Run HuggingFace baseline

```bash
python benchmarks/cosyvoice3/transformers/bench_tts_hf.py \
    --model "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" \
    --num-prompts 50 \
    --gpu-device 0 \
    --result-dir results/
```

### 4) Generate comparison plots

```bash
python benchmarks/cosyvoice3/plot_results.py \
    --results results/bench_async_chunk_*.json results/bench_hf_transformers_*.json \
    --labels "vllm-omni" "hf_transformers" \
    --output results/comparison.png
```

## Batch-size presets

The bench script loads the bundled production deploy (`vllm_omni/deploy/cosyvoice3.yaml`) and layers per-stage budgets on top via `--stage-overrides`, driven by the `BATCH_SIZE` env var. Each batch size picks compatible per-stage `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization` defaults:

| `BATCH_SIZE` | Description |
|:--:|-------------|
| `1` (default) | Single-request processing (lowest latency) |
| `4`  | Moderate-throughput concurrent processing |
| `16` | High-throughput concurrent processing |

The 2-stage pipeline (Talker -> Code2Wav) runs with `async_chunk` streaming enabled via the prod deploy; the `SharedMemoryConnector` streams codec frames (25-frame chunks with 25-frame context overlap) between stages.

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second
