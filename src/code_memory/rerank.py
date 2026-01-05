from __future__ import annotations

import math
import time
from typing import List

from .config import DEFAULT_TOP_K, DEFAULT_TOP_P, FTS_BONUS, OVERSAMPLE_K, PRIORITY_WEIGHT, RECENCY_WEIGHT


def clamp_top_k(value: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, DEFAULT_TOP_K)


def clamp_top_p(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return DEFAULT_TOP_P
    if v <= 0:
        return 1.0
    return min(1.0, v)


def effective_k(limit: int) -> int:
    try:
        mult = max(1, int(OVERSAMPLE_K))
    except Exception:
        mult = 4
    return max(1, int(limit) * mult)


def parse_timestamp(value: str) -> float:
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0.0


def apply_recency_filter(results: List[dict], top_p: float) -> List[dict]:
    if not results:
        return results
    top_p = clamp_top_p(top_p)
    if top_p >= 1.0:
        return results
    sorted_by_time = sorted(results, key=lambda r: parse_timestamp(r.get("created_at", "")), reverse=True)
    keep = max(1, int(math.ceil(len(sorted_by_time) * top_p)))
    keep_ids = {r["id"] for r in sorted_by_time[:keep]}
    return [r for r in results if r["id"] in keep_ids]


def rerank_results(results: List[dict], top_p: float) -> List[dict]:
    if not results:
        return results
    times = [parse_timestamp(r.get("created_at", "")) for r in results]
    now = max(times) if times else time.time()
    max_age = max((now - t) for t in times) if times else 0.0

    for r in results:
        base = r.get("score", 1.0) or 1.0
        if r.get("fts_hit"):
            base -= FTS_BONUS
        priority = r.get("priority") or 3
        base += PRIORITY_WEIGHT * (priority - 1)
        age = now - parse_timestamp(r.get("created_at", ""))
        if max_age > 0:
            base += RECENCY_WEIGHT * (age / max_age)
        r["score"] = base

    results = apply_recency_filter(results, top_p)
    results.sort(key=lambda x: x["score"])
    return results

