"""
Retrieval evaluation pipeline.

Selects 10 stratified queries from data/evaluation.json, runs DINOv3+cosine
retrieval, computes Rank-1/5/10/20 accuracy, prints a live terminal trace,
and saves a markdown report to reports/.

Usage:
    uv run python retrieval_pipeline.py
"""
import json
import random
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from catalogue import embed_image
from recommender import search

EVALUATION_PATH = Path("data/evaluation.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
METADATA_PATH = Path("data/index_metadata.json")
REPORTS_DIR = Path("reports")

SEED = 42
N_QUERIES = 10
RANK_KS = [1, 5, 10, 20]


def select_queries(evaluation: list[dict], n: int, seed: int) -> list[dict]:
    """
    Select n queries from evaluation, stratified by category_name.
    Picks 1 item from each of the n most-represented categories.
    Fixed seed for reproducibility.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list] = defaultdict(list)
    for item in evaluation:
        by_cat[item["category_name"]].append(item)

    top_cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))[:n]
    return [rng.choice(by_cat[cat]) for cat in top_cats]


def build_pair_id_lookup(index_meta: list[dict]) -> dict[int, str]:
    """Map pair_id → catalogue filename from index_metadata list."""
    return {item["pair_id"]: item["filename"] for item in index_meta}


def rank_at_k(results: list[dict], target_pair_id: int, k: int) -> bool:
    """Return True if target_pair_id appears in the top-k results."""
    return target_pair_id in [r["metadata"]["pair_id"] for r in results[:k]]


def run_query(
    query_item: dict,
    embeddings_path: Path,
    metadata_path: Path,
    query_num: int,
    total: int,
) -> dict:
    """Run retrieval for one query. Prints live trace. Returns result dict."""
    sep = "─" * 56
    print(f"\n{sep}")
    print(f"Query {query_num}/{total} | {query_item['category_name']} | {query_item['archetype']}")
    print(f"  [IMAGE]  {query_item['path']}")

    img = Image.open(query_item["path"]).convert("RGB")

    t0 = time.perf_counter()
    try:
        from rembg import remove
        img_rgb = remove(img).convert("RGB")
        rembg_used = True
    except Exception:
        img_rgb = img
        rembg_used = False

    vec = embed_image(img_rgb)
    encode_time = time.perf_counter() - t0
    print(f"  [ENCODE] 384-dim vector in {encode_time:.2f}s (rembg: {'yes' if rembg_used else 'no'})")

    t1 = time.perf_counter()
    results = search(vec, embeddings_path, metadata_path, top_n=max(RANK_KS))
    search_time = time.perf_counter() - t1
    print(f"  [SEARCH] Top-{max(RANK_KS)} retrieved in {search_time:.2f}s")

    target = query_item["pair_id"]
    hits = {k: rank_at_k(results, target, k) for k in RANK_KS}
    top_score = results[0]["score"] if results else 0.0

    parts = " | ".join(f"Rank-{k} {'✓' if hits[k] else '✗'}" for k in RANK_KS)
    print(f"  [RESULT] {parts}  (score {top_score:.3f})")

    return {
        "query_num": query_num,
        "category": query_item["category_name"],
        "archetype": query_item["archetype"],
        "hits": hits,
        "top_score": top_score,
        "encode_time": encode_time,
        "rembg_used": rembg_used,
    }


def generate_report(results: list[dict], index_size: int) -> str:
    """Render the markdown evaluation report."""
    today = date.today().isoformat()
    lines = [
        f"# Retrieval Evaluation — {today}",
        "",
        f"**Index:** {index_size:,} items | **Queries:** {len(results)} | **Model:** dinov3-vits16",
        "",
        "---",
        "",
        "## Per-query results",
        "",
    ]

    for r in results:
        rank_str = " | ".join(
            f"Rank-{k} {'✓' if r['hits'][k] else '✗'}" for k in RANK_KS
        )
        lines += [
            f"### Query {r['query_num']} — {r['category']} | {r['archetype']}",
            f"- {rank_str}",
            f"- Top-1 score: {r['top_score']:.3f}",
            f"- Encode time: {r['encode_time']:.2f}s (rembg: {'yes' if r['rembg_used'] else 'no'})",
            "",
        ]

    lines += [
        "---",
        "",
        "## Summary",
        "",
        "| Query | Category | Archetype | Rank-1 | Rank-5 | Rank-10 | Rank-20 | Top-1 Score |",
        "|-------|----------|-----------|--------|--------|---------|---------|-------------|",
    ]

    for r in results:
        row = (
            f"| {r['query_num']} "
            f"| {r['category']} "
            f"| {r['archetype']} "
            f"| {'✓' if r['hits'][1] else '✗'} "
            f"| {'✓' if r['hits'][5] else '✗'} "
            f"| {'✓' if r['hits'][10] else '✗'} "
            f"| {'✓' if r['hits'][20] else '✗'} "
            f"| {r['top_score']:.3f} |"
        )
        lines.append(row)

    means = {k: sum(r["hits"][k] for r in results) / len(results) for k in RANK_KS}
    mean_score = sum(r["top_score"] for r in results) / len(results)
    lines.append(
        f"| **Mean** | | "
        f"| **{means[1] * 100:.0f}%** "
        f"| **{means[5] * 100:.0f}%** "
        f"| **{means[10] * 100:.0f}%** "
        f"| **{means[20] * 100:.0f}%** "
        f"| **{mean_score:.3f}** |"
    )

    lines += [
        "",
        "---",
        "",
        "## Analysis",
        "",
        f"Overall Rank-1 accuracy: {means[1] * 100:.0f}% across {len(results)} queries. "
        f"Rank-20 accuracy: {means[20] * 100:.0f}%.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        print("Index not found. Run: uv run python index.py")
        raise SystemExit(1)

    with open(EVALUATION_PATH) as f:
        evaluation = json.load(f)
    with open(METADATA_PATH) as f:
        index_meta = json.load(f)

    queries = select_queries(evaluation, N_QUERIES, SEED)
    pair_id_lookup = build_pair_id_lookup(index_meta)

    for q in queries:
        if q["pair_id"] not in pair_id_lookup:
            print(f"[warn] pair_id {q['pair_id']} not found in catalogue index")

    query_results = []
    for i, q in enumerate(queries, 1):
        result = run_query(q, EMBEDDINGS_PATH, METADATA_PATH, i, len(queries))
        query_results.append(result)

    report = generate_report(query_results, len(index_meta))

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"retrieval_eval_{date.today().isoformat()}.md"
    report_path.write_text(report)

    print(f"\n{'=' * 56}")
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()