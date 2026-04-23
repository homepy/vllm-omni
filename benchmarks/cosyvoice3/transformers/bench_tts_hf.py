"""Benchmark CosyVoice3 using HuggingFace transformers (vllm-omni offline inference).

Measures E2E latency, RTF, and audio duration for offline (non-serving) inference.
Results are saved in the same JSON format as bench_tts_serve.py for unified plotting.

Usage:
    python bench_tts_hf.py \
        --model pretrained_models/Fun-CosyVoice3-0.5B \
        --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
        --num-prompts 50 \
        --num-warmups 3 \
        --gpu-device 0 \
        --result-dir results/
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer

PROMPTS = [
    "你好，这是一段语音合成的测试文本。",
    "今天天气真好，我们一起去公园散步吧。",
    "学习新技术需要耐心和持续的练习。",
    "收到好友从远方寄来的生日礼物，那份意外的惊喜让我热泪盈眶。",
    "人工智能技术正在改变我们的生活方式。",
    "愿你每天都能保持积极乐观的心态面对生活。",
    "音乐是人类共同的语言，能够跨越文化和语言的障碍。",
    "健康的身体是幸福生活的重要基础。",
    "书籍是人类进步的阶梯，让我们一起多读书读好书。",
    "科技创新推动社会不断向前发展进步。",
    "春天来了，万物复苏，大地一片生机勃勃的景象。",
    "旅行能够开阔视野，增长见识，丰富人生阅历。",
]

# Reference audio URL and text for voice cloning
REF_AUDIO_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
REF_TEXT = "希望你以后能够做的比我还好呦。"
REF_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>Testing my voices. Why should I not?"

SAMPLE_RATE = 24000  # Output sample rate from vllm-omni CosyVoice3


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 1  # always 1 for offline
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats - not applicable for HF offline, set to E2E for compatibility
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    # Per-request details
    per_request: list = field(default_factory=list)


def load_reference_audio(url: str) -> tuple[np.ndarray, int]:
    """Load reference audio from URL."""
    import io
    from urllib.request import urlopen

    with urlopen(url, timeout=30) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def concat_audio(audio_val) -> np.ndarray:
    """Concatenate audio tensors into a single numpy array."""
    if isinstance(audio_val, list):
        tensors = []
        for t in audio_val:
            if t is None:
                continue
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "float"):
                t = t.float()
            if isinstance(t, torch.Tensor):
                tensors.append(t.reshape(-1))
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)

    if hasattr(audio_val, "detach"):
        audio_val = audio_val.detach()
    if hasattr(audio_val, "cpu"):
        audio_val = audio_val.cpu()
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float()
    if hasattr(audio_val, "numpy"):
        audio_val = audio_val.numpy()
    audio_np = np.asarray(audio_val, dtype=np.float32)
    return audio_np.reshape(-1)


def get_sampling_params(tokenizer, prompt: str, config: CosyVoice3Config) -> SamplingParams:
    """Build SamplingParams for CosyVoice3 based on config."""
    sampling_cfg = config.llm.get("sampling", {})
    eos_token_id = int(config.llm["eos_token_id"])
    text_len = max(1, len(tokenizer.encode(prompt, allowed_special=config.allowed_special)))
    return SamplingParams(
        temperature=1.0,
        top_p=float(sampling_cfg.get("top_p", 0.8)),
        top_k=int(sampling_cfg.get("top_k", 25)),
        repetition_penalty=2.0,
        stop_token_ids=[eos_token_id],
        min_tokens=int(text_len * config.min_token_text_ratio),
        max_tokens=int(text_len * config.max_token_text_ratio),
        detokenize=False,
    )


def run_benchmark(args):
    from vllm_omni.entrypoints.omni import Omni

    device = f"cuda:{args.gpu_device}"
    print(f"Loading model: {args.model} on {device}")
    
    tokenizer_path = args.tokenizer
    if not tokenizer_path:
        tokenizer_path = str(Path(args.model) / "CosyVoice-BlankEN")
    
    omni = Omni(
        model=args.model,
        trust_remote_code=True,
        tokenizer=tokenizer_path,
        log_stats=True,
    )
    print("Model loaded.")

    # Load reference audio
    print(f"Loading reference audio from {REF_AUDIO_URL}...")
    ref_audio, ref_sr = load_reference_audio(REF_AUDIO_URL)
    ref_audio_data = (ref_audio.astype(np.float32), ref_sr)
    print(f"Reference audio loaded: {len(ref_audio)} samples at {ref_sr} Hz")

    # Build prompt list
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]

    # Warmup
    if args.num_warmups > 0:
        print(f"Warming up with {args.num_warmups} requests...")
        for i in range(args.num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            config = CosyVoice3Config()
            tokenizer = get_qwen_tokenizer(
                token_path=tokenizer_path,
                skip_special_tokens=config.skip_special_tokens,
                version=config.version,
            )
            sampling_params_list = [
                get_sampling_params(tokenizer, prompt, config),
                SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=256, detokenize=False),
            ]
            request_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": ref_audio_data},
                    "mm_processor_kwargs": {"prompt_text": REF_PROMPT_TEXT, "sample_rate": ref_sr},
                }
            ]
            list(omni.generate(request_inputs, sampling_params_list=sampling_params_list))
        torch.cuda.synchronize(device)
        print("Warmup done.")

    # Benchmark
    print(f"Running {args.num_prompts} requests sequentially...")
    e2e_times = []
    rtfs = []
    audio_durations = []
    per_request = []
    failed = 0

    config = CosyVoice3Config()
    tokenizer = get_qwen_tokenizer(
        token_path=tokenizer_path,
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )

    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        try:
            torch.cuda.synchronize(device)
            st = time.perf_counter()

            sampling_params_list = [
                get_sampling_params(tokenizer, prompt, config),
                SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=256, detokenize=False),
            ]
            request_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": ref_audio_data},
                    "mm_processor_kwargs": {"prompt_text": REF_PROMPT_TEXT, "sample_rate": ref_sr},
                }
            ]
            outputs = list(omni.generate(request_inputs, sampling_params_list=sampling_params_list))

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - st

            # Extract audio from output
            audio = None
            sr = SAMPLE_RATE
            if outputs and outputs[0].multimodal_output:
                mm = outputs[0].multimodal_output
                if "audio" in mm:
                    audio = concat_audio(mm["audio"])
                    if "sr" in mm:
                        sr = mm["sr"]
                        if isinstance(sr, list) and sr:
                            sr = sr[-1]
                        if hasattr(sr, "item"):
                            sr = sr.item()
                        sr = int(sr)

            if audio is None or len(audio) == 0:
                raise ValueError("No audio output generated")

            audio_dur = len(audio) / sr
            rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

            e2e_times.append(elapsed)
            rtfs.append(rtf)
            audio_durations.append(audio_dur)
            per_request.append(
                {
                    "e2e_ms": elapsed * 1000,
                    "ttfp_ms": elapsed * 1000,  # no streaming, TTFP = E2E
                    "rtf": rtf,
                    "audio_duration_s": audio_dur,
                    "prompt": prompt,
                }
            )

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{args.num_prompts}] e2e={elapsed * 1000:.0f}ms  rtf={rtf:.3f}  audio={audio_dur:.2f}s")

        except Exception as e:
            print(f"  [{i + 1}/{args.num_prompts}] FAILED: {e}")
            failed += 1

    total_duration = time.perf_counter() - total_start
    completed = len(e2e_times)

    omni.close()

    # Compute stats
    result = BenchmarkResult(
        config_name=args.config_name,
        concurrency=1,
        num_prompts=args.num_prompts,
        completed=completed,
        failed=failed,
        duration_s=total_duration,
    )

    if e2e_times:
        e2e_ms = [t * 1000 for t in e2e_times]

        result.mean_e2e_ms = float(np.mean(e2e_ms))
        result.median_e2e_ms = float(np.median(e2e_ms))
        result.std_e2e_ms = float(np.std(e2e_ms))
        result.p90_e2e_ms = float(np.percentile(e2e_ms, 90))
        result.p95_e2e_ms = float(np.percentile(e2e_ms, 95))
        result.p99_e2e_ms = float(np.percentile(e2e_ms, 99))

        # For offline, TTFP = E2E (no streaming)
        result.mean_ttfp_ms = result.mean_e2e_ms
        result.median_ttfp_ms = result.median_e2e_ms
        result.std_ttfp_ms = result.std_e2e_ms
        result.p90_ttfp_ms = result.p90_e2e_ms
        result.p95_ttfp_ms = result.p95_e2e_ms
        result.p99_ttfp_ms = result.p99_e2e_ms

        result.mean_rtf = float(np.mean(rtfs))
        result.median_rtf = float(np.median(rtfs))
        result.std_rtf = float(np.std(rtfs))
        result.p99_rtf = float(np.percentile(rtfs, 99))

        result.mean_audio_duration_s = float(np.mean(audio_durations))
        result.total_audio_duration_s = float(np.sum(audio_durations))
        result.audio_throughput = result.total_audio_duration_s / total_duration
        result.request_throughput = completed / total_duration
        result.per_request = per_request

    # Print summary in standardized performance template
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{completed:<10}")
    print(f"{'Failed requests:':<40}{failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{1:<10}")
    print(f"{'Benchmark duration (s):':<40}{total_duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{result.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{result.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{result.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{result.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{result.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{result.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{result.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{result.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{result.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{result.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{result.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{result.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    # Save results (as a list with single concurrency=1 entry, matching serve format)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump([asdict(result)], f, indent=2)
    print(f"Results saved to {result_file}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 HuggingFace Benchmark")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to CosyVoice3 model directory"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Path to tokenizer directory (default: <model>/CosyVoice-BlankEN)"
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--config-name", type=str, default="hf_transformers", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
