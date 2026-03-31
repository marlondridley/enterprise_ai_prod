import json
import os
from statistics import mean
from typing import Any


BASE_DIR = os.path.dirname(__file__)
RUNS_DIR = os.path.join(BASE_DIR, "store", "runs")
BASELINES_DIR = os.path.join(BASE_DIR, "store", "baselines")


def ensure_eval_dirs() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(BASELINES_DIR, exist_ok=True)


def save_run_results(results: list[dict[str, Any]], filename: str) -> str:
    ensure_eval_dirs()
    path = os.path.join(RUNS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return path


def save_baseline(results: list[dict[str, Any]], filename: str = "baseline.json") -> str:
    ensure_eval_dirs()
    path = os.path.join(BASELINES_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return path


def load_baseline(filename: str = "baseline.json") -> list[dict[str, Any]]:
    path = os.path.join(BASELINES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline file not found: {filename}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "avg_llm_score": 0.0,
            "deterministic_pass_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_total_tokens": 0.0,
        }

    return {
        "avg_llm_score": mean(r["llm_score"] for r in results),
        "deterministic_pass_rate": mean(
            1.0 if r["deterministic_pass"] else 0.0 for r in results
        ),
        "avg_latency_ms": mean(r["latency_ms"] for r in results),
        "avg_total_tokens": mean(r["total_tokens"] for r in results),
    }


def compare_to_baseline(
    current_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    *,
    max_score_drop: float = 0.20,
    max_pass_rate_drop: float = 0.05,
    max_latency_increase_pct: float = 0.25,
    max_token_increase_pct: float = 0.25,
) -> dict[str, Any]:
    current = summarize_results(current_results)
    baseline = summarize_results(baseline_results)

    failures: list[str] = []

    if current["avg_llm_score"] < (baseline["avg_llm_score"] - max_score_drop):
        failures.append(
            f"Average LLM score regressed from {baseline['avg_llm_score']:.2f} "
            f"to {current['avg_llm_score']:.2f}"
        )

    if current["deterministic_pass_rate"] < (
        baseline["deterministic_pass_rate"] - max_pass_rate_drop
    ):
        failures.append(
            f"Deterministic pass rate regressed from {baseline['deterministic_pass_rate']:.2%} "
            f"to {current['deterministic_pass_rate']:.2%}"
        )

    if baseline["avg_latency_ms"] > 0:
        latency_growth = (
            current["avg_latency_ms"] - baseline["avg_latency_ms"]
        ) / baseline["avg_latency_ms"]
        if latency_growth > max_latency_increase_pct:
            failures.append(
                f"Average latency increased from {baseline['avg_latency_ms']:.1f} ms "
                f"to {current['avg_latency_ms']:.1f} ms"
            )

    if baseline["avg_total_tokens"] > 0:
        token_growth = (
            current["avg_total_tokens"] - baseline["avg_total_tokens"]
        ) / baseline["avg_total_tokens"]
        if token_growth > max_token_increase_pct:
            failures.append(
                f"Average total tokens increased from {baseline['avg_total_tokens']:.1f} "
                f"to {current['avg_total_tokens']:.1f}"
            )

    return {
        "passed": len(failures) == 0,
        "current_summary": current,
        "baseline_summary": baseline,
        "failures": failures,
    }


def fail_if_regressed(comparison: dict[str, Any]) -> None:
    if comparison["passed"]:
        return

    message = "\n".join(comparison["failures"])
    raise RuntimeError(f"Regression gate failed:\n{message}")