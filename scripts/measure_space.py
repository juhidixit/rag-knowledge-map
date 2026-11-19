"""Measure the deployed Space: cold start and warm query latency, as a client sees them.

Cold start = time from a restart request until the app serves its page with the preloaded papers
indexed (model download + preload included; Spaces have no persistent disk, so waking from sleep
does the same work). Warm latency = round-trip time of /api/search from the runner, plus the
server-side retrieval time the API reports. Writes JSON for the README.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone

QUESTIONS = [
    "What problem does this paper address?",
    "What datasets are used in the evaluation?",
    "What are the main results?",
    "What baselines do they compare against?",
    "What are the limitations of the approach?",
    "How is retrieval performed?",
    "Which language models are used?",
    "How is the method evaluated?",
    "What is the main contribution?",
    "What future work do the authors suggest?",
]


def get_json(url: str, data: dict | None = None, timeout: float = 60) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def wait_ready(base: str, deadline_s: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(base + "/", timeout=30) as r:
                page_ok = r.status == 200 and b"ragmap" in r.read()
            if page_ok and get_json(base + "/api/health")["documents"] > 0:
                return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"Space not ready after {deadline_s:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="juhidixit/ragmap-demo")
    ap.add_argument("--rounds", type=int, default=3, help="passes over the question list")
    ap.add_argument("--cold", action="store_true", help="restart the Space first and time the cold start")
    ap.add_argument("--out", default="results/space_latency.json")
    args = ap.parse_args()
    base = "https://" + args.space.replace("/", "-").replace("_", "-").lower() + ".hf.space"

    cold = None
    if args.cold:
        from huggingface_hub import HfApi

        HfApi(token=os.environ["HF_TOKEN"]).restart_space(args.space)
        time.sleep(5)  # let the old container stop answering
        cold = wait_ready(base, 1200)
        print(f"cold start: {cold:.1f} s")
    else:
        wait_ready(base, 1200)

    docs = get_json(base + "/api/documents")
    rtt, server = [], []
    get_json(base + "/api/search", {"query": QUESTIONS[0], "answer": True})  # warm-up, not counted
    for _ in range(args.rounds):
        for q in QUESTIONS:
            for d in docs:
                t0 = time.perf_counter()
                res = get_json(base + "/api/search", {"query": q, "doc_ids": [d["doc_id"]], "answer": True})
                rtt.append((time.perf_counter() - t0) * 1000)
                server.append(res["timings"].get("total_ms", sum(res["timings"].values())))

    out = {
        "space": args.space,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client": "GitHub Actions ubuntu-latest runner",
        "mode": "hybrid_rerank, k=5, extractive answer",
        "documents": len(docs),
        "queries": len(rtt),
        "cold_start_s": round(cold, 1) if cold is not None else None,
        "round_trip_ms": {"p50": round(statistics.median(rtt)), "p95": round(pct(rtt, 0.95))},
        "server_ms": {"p50": round(statistics.median(server)), "p95": round(pct(server, 0.95))},
    }
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
