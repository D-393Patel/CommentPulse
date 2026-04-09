import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 4)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(float(ordered[lower]), 4)
    weight = position - lower
    interpolated = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(float(interpolated), 4)


def build_payload(comments_per_request: int) -> dict:
    comments = []
    for index in range(comments_per_request):
        comments.append(
            {
                "text": f"Load test comment {index} with useful tutorial feedback and timestamp coverage.",
                "timestamp": f"2026-03-31T00:{index % 60:02d}:00Z",
            }
        )
    return {"comments": comments}


def run_request(base_url: str, comments_per_request: int, timeout_seconds: int) -> dict:
    payload = build_payload(comments_per_request)
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict_with_timestamps",
            json=payload,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "latency_ms": round(elapsed * 1000, 4),
            "response_size": len(response.content),
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": round(elapsed * 1000, 4),
            "error": str(error),
            "response_size": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight concurrent load test against the local API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base URL of the running API.")
    parser.add_argument("--requests", type=int, default=50, help="Total number of requests to send.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers.")
    parser.add_argument("--comments-per-request", type=int, default=25, help="Comments per prediction request.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Per-request timeout.")
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "performance", "load_test_report.json"),
        help="Where to write the JSON load test report.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_request, args.base_url, args.comments_per_request, args.timeout_seconds)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_duration = time.perf_counter() - started
    latencies = [result["latency_ms"] for result in results]
    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]

    report = {
        "base_url": args.base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "comments_per_request": args.comments_per_request,
        "total_duration_seconds": round(total_duration, 4),
        "throughput_requests_per_second": round(args.requests / total_duration, 4) if total_duration else 0.0,
        "success_count": len(successes),
        "failure_count": len(failures),
        "error_rate": round(len(failures) / args.requests, 4) if args.requests else 0.0,
        "latency_ms": {
            "min": round(min(latencies), 4) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 4) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "status_codes": {
            str(code): sum(1 for result in results if result.get("status_code") == code)
            for code in sorted({result.get("status_code") for result in results})
        },
        "sample_failures": failures[:5],
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
