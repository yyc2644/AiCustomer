"""
多用户并发压测 deepseek-r1:1.5b（基于 test/chattest1.py 的单用户脚本）。

使用说明（在项目根目录执行）:
    python -m test.benchmark_ollama_concurrent \
        --users 10 --requests-per-user 5 \
        --prompt "你是谁"

注意：并发数和总请求数不要一开始就开太大，先小规模试跑。
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests

OLLAMA_URL = "http://192.168.4.29:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"


def call_ollama_once(prompt: str, timeout: float = 120.0) -> Dict:
    """调用一次 Ollama，返回统计信息，不打印模型输出。"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
    }

    start = time.perf_counter()
    response = requests.post(
        OLLAMA_URL, json=payload, stream=True, timeout=timeout
    )
    response.raise_for_status()

    final_stats: Dict = {}
    # 这里不关心具体文本，只消费流以拿到最后一块统计
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("done"):
            final_stats = data
            break

    end = time.perf_counter()
    duration = end - start

    # 从 Ollama 返回中提取 token 统计信息
    prompt_tokens = final_stats.get("prompt_eval_count", 0)
    completion_tokens = final_stats.get("eval_count", 0)

    return {
        "ok": True,
        "latency": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def worker(user_id: int, n_requests: int, prompt: str, results: List[Dict], lock: threading.Lock) -> None:
    """单个“用户”连续发起多次请求。"""
    for i in range(n_requests):
        try:
            stat = call_ollama_once(prompt)
        except Exception as e:  # noqa: BLE001
            stat = {"ok": False, "error": repr(e), "latency": None}
        stat["user_id"] = user_id
        stat["seq"] = i
        with lock:
            results.append(stat)


def run_benchmark(users: int, requests_per_user: int, prompt: str, max_workers: int | None = None) -> None:
    total_requests = users * requests_per_user
    print(f"并发用户数: {users}, 每个用户请求数: {requests_per_user}, 总请求数: {total_requests}")
    print(f"测试 Prompt: {prompt!r}")

    results: List[Dict] = []
    lock = threading.Lock()

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers or users) as executor:
        futures = [
            executor.submit(worker, user_id, requests_per_user, prompt, results, lock)
            for user_id in range(users)
        ]
        # 等待所有用户完成
        for f in as_completed(futures):
            _ = f.result()
    end_all = time.perf_counter()

    duration_all = end_all - start_all
    print(f"\n整体耗时: {duration_all:.2f}s")

    # 统计结果
    ok_results = [r for r in results if r.get("ok")]
    fail_results = [r for r in results if not r.get("ok")]

    print(f"成功请求数: {len(ok_results)}, 失败请求数: {len(fail_results)}")
    if fail_results:
        # 仅打印前几条错误，避免刷屏
        print("部分错误示例（前 5 条）：")
        for r in fail_results[:5]:
            print(f"  user={r.get('user_id')} seq={r.get('seq')} error={r.get('error')}")

    if not ok_results:
        print("全部请求失败，无法统计性能。")
        return

    latencies = [r["latency"] for r in ok_results if r.get("latency") is not None]
    total_tokens = [r["total_tokens"] for r in ok_results]

    print("\n=== 延迟统计 ===")
    print(f"平均延迟: {statistics.mean(latencies):.2f}s")
    print(f"50% 分位(P50): {statistics.median(latencies):.2f}s")
    print(f"最大延迟: {max(latencies):.2f}s, 最小延迟: {min(latencies):.2f}s")

    print("\n=== 吞吐量统计 ===")
    print(f"QPS(请求/秒): {len(ok_results) / duration_all:.2f}")
    print(
        f"Token 吞吐量: {sum(total_tokens) / duration_all:.2f} tokens/s "
        f"(平均每次 {statistics.mean(total_tokens):.1f} tokens)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多用户并发压测本地 Ollama deepseek-r1:1.5b 性能")
    parser.add_argument("--users", type=int, default=10, help="并发用户数")
    parser.add_argument("--requests-per-user", type=int, default=1, help="每个用户连续请求次数")
    parser.add_argument("--prompt", type=str, default="你是谁", help="测试使用的提问内容")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="线程池大小（默认等于 users）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        users=args.users,
        requests_per_user=args.requests_per_user,
        prompt=args.prompt,
        max_workers=args.workers,
    )


