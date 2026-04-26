"""
Recall@K evaluation — Marqo FashionSigLIP on DeepFashion InShop.

Protocol:
- Gallery  : 12,861 front-view images indexed in data/embeddings_marqo_fashion_siglip.npy
- Queries  : non-front views on disk (side, back, additional) for items present in the gallery
- Ground truth: any gallery entry sharing the same item ID (id_XXXXXXXX) as the query
- Metric   : Recall@K = % of queries where ≥1 of the top-K results is the correct item

Usage:
    uv run python -m backend.scripts.evaluate_recall
    uv run python -m backend.scripts.evaluate_recall --n-queries 200 --k-values 1 3 5 10
    uv run python -m backend.scripts.evaluate_recall --all-queries
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.embedder import embed_image

EMBEDDINGS_PATH = Path("data/embeddings_marqo_fashion_siglip.npy")
METADATA_PATH = Path("data/index_metadata.json")
DATASET_ROOT = Path("dataset/deepfashion-inshop/img_highres")
RESULTS_PATH = Path("docs/Embedder/recall_evaluation.json")

DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_N_QUERIES = 500
SEED = 42


def extract_item_id(path: str) -> str | None:
    m = re.search(r"id_\d+", path)
    return m.group() if m else None


def build_gallery_index(metadata: list[dict]) -> dict[str, list[int]]:
    """Map item_id → list of gallery indices."""
    index: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(metadata):
        item_id = extract_item_id(entry["path"])
        if item_id:
            index[item_id].append(i)
    return dict(index)


def collect_query_paths(gallery_ids: set[str], n_queries: int | None, seed: int = SEED) -> list[tuple[Path, str]]:
    """
    Return (image_path, item_id) for non-front views whose item_id is in the gallery.
    Shuffled and limited to n_queries (None = all).
    """
    queries = []
    for img_path in DATASET_ROOT.rglob("*.jpg"):
        if "_1_front" in img_path.name:
            continue
        item_id = extract_item_id(str(img_path))
        if item_id and item_id in gallery_ids:
            queries.append((img_path, item_id))

    random.seed(seed)
    random.shuffle(queries)
    if n_queries is not None:
        queries = queries[:n_queries]
    return queries


def recall_at_k(
    query_vec: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_index: dict[str, list[int]],
    item_id: str,
    k: int,
) -> bool:
    """Return True if the correct item appears in the top-K results."""
    sims = gallery_embs @ query_vec
    top_k_indices = set(np.argsort(sims)[::-1][:k].tolist())
    correct_indices = set(gallery_index.get(item_id, []))
    return bool(top_k_indices & correct_indices)


def run_evaluation(
    gallery_embs: np.ndarray,
    metadata: list[dict],
    gallery_index: dict[str, list[int]],
    queries: list[tuple[Path, str]],
    k_values: list[int],
) -> dict:
    hits: dict[int, int] = {k: 0 for k in k_values}
    category_hits: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in k_values})
    category_total: dict[str, int] = defaultdict(int)
    errors = 0

    for img_path, item_id in tqdm(queries, desc="Evaluating", unit="query"):
        # Retrieve category from gallery metadata
        gallery_entry = metadata[gallery_index[item_id][0]]
        category = gallery_entry.get("category_name", "unknown")

        try:
            img = Image.open(img_path).convert("RGB")
            query_vec = embed_image(img)
        except Exception as e:
            tqdm.write(f"[warn] {img_path.name}: {e}")
            errors += 1
            continue

        category_total[category] += 1
        for k in k_values:
            if recall_at_k(query_vec, gallery_embs, gallery_index, item_id, k):
                hits[k] += 1
                category_hits[category][k] += 1

    n = len(queries) - errors
    overall = {f"Recall@{k}": round(hits[k] / n, 4) if n else 0.0 for k in k_values}
    per_category = {}
    for cat, total in sorted(category_total.items()):
        per_category[cat] = {
            f"Recall@{k}": round(category_hits[cat][k] / total, 4) if total else 0.0
            for k in k_values
        }
        per_category[cat]["n_queries"] = total

    return {
        "n_queries": n,
        "errors": errors,
        "k_values": k_values,
        "overall": overall,
        "per_category": per_category,
    }


def print_results(results: dict) -> None:
    k_values = results["k_values"]
    print(f"\n{'='*55}")
    print(f"  Recall@K — Marqo FashionSigLIP  ({results['n_queries']} queries)")
    print(f"{'='*55}")
    header = "Category".ljust(22) + "  n  " + "  ".join(f"@{k:2d}" for k in k_values)
    print(header)
    print("-" * len(header))
    for cat, vals in results["per_category"].items():
        n = vals["n_queries"]
        scores = "  ".join(f"{vals[f'Recall@{k}']:.2%}" for k in k_values)
        print(f"{cat:<22} {n:>3}  {scores}")
    print("-" * len(header))
    overall = results["overall"]
    scores = "  ".join(f"{overall[f'Recall@{k}']:.2%}" for k in k_values)
    print(f"{'OVERALL':<22} {results['n_queries']:>3}  {scores}")
    print(f"{'='*55}")
    if results["errors"]:
        print(f"  ({results['errors']} images skipped due to errors)")


def write_markdown_report(results: dict, args: argparse.Namespace, metadata: list[dict]) -> str:
    gallery_embs = np.load(EMBEDDINGS_PATH).astype(np.float32)

    lines = [
        "---",
        f"gallery_items: {gallery_embs.shape[0]}",
        f"gallery_dims: {gallery_embs.shape[1]}",
        f"gallery_model: Marqo FashionSigLIP",
        f"n_queries: {results['n_queries']}",
        f"query_types: non-front views (side, back, additional)",
        f"dataset: DeepFashion InShop",
        f"k_values: {results['k_values']}",
        f"seed: {args.seed}",
        f"errors: {results['errors']}",
        "---",
        "",
        "# Recall@K Evaluation Report",
        "",
        "## About this metric",
        "",
        "**Recall@K** measures the ability of a retrieval system to find at least one relevant item",
        "within the top K results returned for a given query. It is the fraction of queries for which",
        "the correct item appears among the top-K matches.",
        "",
        "- **Recall@1 = 80%** → the correct item is the top-1 result in 80% of queries.",
        "- **Recall@5 = 95%** → the correct item is in the top-5 in 95% of queries.",
        "- A higher Recall@K is better; a value of 1.0 means perfect retrieval.",
        "- K=1 is the most stringent; K=10 is the most lenient.",
        "",
        "**What is evaluated here:** non-front views (side, back, additional angles) are used as",
        "queries against a gallery of front-view images. Ground truth is any gallery image",
        "sharing the same item ID (`id_XXXXXXXX`). The system must retrieve a matching item",
        "regardless of the view angle, testing view-invariant retrieval.",
        "",
    ]

    k_values = results["k_values"]
    lines.append(f"## Overall — {results['n_queries']} queries")
    lines.append("")
    lines.append("| Metric     | Value  |")
    lines.append("|------------|--------|")
    for k in k_values:
        lines.append(f"| Recall@{k:>2}     | {results['overall'][f'Recall@{k}']:.2%}  |")
    lines.append("")

    lines.append("## Per Category")
    lines.append("")
    lines.append("| Category               |   n   |" + "".join(f" @{k:>2}   |" for k in k_values))
    sep = "|------------|------:|" + "".join("------:|" for _ in k_values)
    lines.append(sep)
    for cat, vals in results["per_category"].items():
        scores = "".join(f" {vals[f'Recall@{k}']:.2%} |" for k in k_values)
        lines.append(f"| {cat:<20} | {vals['n_queries']:>4} |{scores}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@K evaluation for Marqo FashionSigLIP retrieval")
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES,
                        help=f"Number of query images to evaluate (default: {DEFAULT_N_QUERIES})")
    parser.add_argument("--all-queries", action="store_true",
                        help="Use all available non-front query images (slow)")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        metavar="K", help=f"K values for Recall@K (default: {DEFAULT_K_VALUES})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for query sampling (default: {SEED})")
    args = parser.parse_args()

    print("Loading gallery embeddings and metadata...")
    gallery_embs = np.load(EMBEDDINGS_PATH).astype(np.float32)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(f"  Gallery: {gallery_embs.shape[0]:,} items × {gallery_embs.shape[1]} dims")

    gallery_index = build_gallery_index(metadata)
    gallery_ids = set(gallery_index.keys())
    print(f"  Unique items in gallery: {len(gallery_ids):,}")

    n_queries = None if args.all_queries else args.n_queries
    print(f"\nCollecting query images ({'all' if n_queries is None else n_queries})...")
    queries = collect_query_paths(gallery_ids, n_queries, seed=args.seed)
    print(f"  Found {len(queries):,} query images")

    results = run_evaluation(gallery_embs, metadata, gallery_index, queries, sorted(args.k_values))

    print_results(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {RESULTS_PATH}")

    md_path = RESULTS_PATH.with_suffix(".md")
    md_path.write_text(write_markdown_report(results, args, metadata))
    print(f"Markdown report saved → {md_path}")


if __name__ == "__main__":
    main()
