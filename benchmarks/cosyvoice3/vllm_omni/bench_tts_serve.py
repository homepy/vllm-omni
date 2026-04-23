"""Benchmark client for CosyVoice3 via /v1/audio/speech endpoint.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels. Saves results as JSON for plotting.

CosyVoice3 uses voice cloning with reference audio and text.

Usage:
    python bench_tts_serve.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --result-dir results/
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

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
# Official CosyVoice zero-shot prompt audio and its transcript
REF_AUDIO = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
REF_TEXT = "希望你以后能够做的比我还好呦。"
SAMPLE_RATE = 22050


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats (ms)
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
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


def wav_bytes_to_duration(num_bytes: int, sample_rate: int = SAMPLE_RATE, bits_per_sample: int = 16) -> float:
    """Convert WAV byte count to duration in seconds.
    
    WAV header is 44 bytes, after that it's data.
    Data bytes / (channels * bytes_per_sample) / sample_rate = duration
    Assuming mono (1 channel), 16-bit (2 bytes per sample)
    """
    if num_bytes <= 44:
        return 0.0
    data_bytes = num_bytes - 44  # Subtract WAV header
    if data_bytes < 0:
        return 0.0
    bytes_per_sample = bits_per_sample // 8
    num_samples = data_bytes // bytes_per_sample
    return num_samples / sample_rate


def create_payload(prompt: str) -> dict:
    """Create CosyVoice3 request payload with voice cloning parameters."""
    payload = {
        "input": prompt,
        "ref_audio": REF_AUDIO,
        "ref_text": REF_TEXT,
        "stream": True,
        "response_format": "wav",
        "sample_rate": SAMPLE_RATE,
    }
    return payload


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure latency metrics."""
    payload = create_payload(prompt)

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

            first_chunk = True
            total_bytes = 0

            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttf = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = wav_bytes_to_duration(total_bytes)

            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int = 3,
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level."""
    api_url = f"http://{host}:{port}/v1/audio/speech"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    # Warmup
    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = []
        for i in range(num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            warmup_tasks.append(send_tts_request(session, api_url, prompt))
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    # Build request list
    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    # Run benchmark
    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            return await send_tts_request(session, api_url, prompt, pbar)

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    # Compute stats
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttf * 1000 for r in successful]  # convert to ms
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]

        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))
        bench.p99_rtf = float(np.percentile(rtfs, 99))

        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttfp_ms": r.ttf * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    # Print summary in standardized performance template
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench


async def main(args):
    all_results = []

    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    # Save results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts per concurrency level")
    parser.add_argument(
        "--max-concurrency", type=int, nargs="+", default=[1, 4, 10], help="Concurrency levels to test"
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument(
        "--config-name", type=str, default="async_chunk", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
