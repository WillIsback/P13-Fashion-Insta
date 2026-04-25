import json
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


def _make_eval_items():
    cats = [
        "short sleeve top", "short sleeve top", "short sleeve top",
        "long sleeve top", "long sleeve top",
        "trousers", "trousers",
        "skirt",
        "shorts",
        "vest",
        "sling",
        "short sleeve outwear",
        "long sleeve outwear",
        "dress",
    ]
    return [
        {
            "filename": f"{i:06d}.jpg",
            "path": f"/fake/{i:06d}.jpg",
            "pair_id": i,
            "category_name": cat,
            "archetype": "Casual chic",
            "fashn_category": "tops",
            "b_box": [],
            "split_source": "validation",
        }
        for i, cat in enumerate(cats)
    ]


EVAL_ITEMS = _make_eval_items()


def test_select_queries_returns_n_items():
    from retrieval_pipeline import select_queries
    queries = select_queries(EVAL_ITEMS, n=10, seed=42)
    assert len(queries) == 10


def test_select_queries_covers_top_categories():
    from retrieval_pipeline import select_queries
    queries = select_queries(EVAL_ITEMS, n=10, seed=42)
    cats = {q["category_name"] for q in queries}
    assert "short sleeve top" in cats
    assert "long sleeve top" in cats


def test_select_queries_is_reproducible():
    from retrieval_pipeline import select_queries
    q1 = select_queries(EVAL_ITEMS, n=5, seed=42)
    q2 = select_queries(EVAL_ITEMS, n=5, seed=42)
    assert [q["pair_id"] for q in q1] == [q["pair_id"] for q in q2]


def test_build_pair_id_lookup():
    from retrieval_pipeline import build_pair_id_lookup
    meta = [{"pair_id": 1, "filename": "a.jpg"}, {"pair_id": 2, "filename": "b.jpg"}]
    lookup = build_pair_id_lookup(meta)
    assert lookup == {1: "a.jpg", 2: "b.jpg"}


def test_rank_at_k_hit():
    from retrieval_pipeline import rank_at_k
    results = [
        {"metadata": {"pair_id": 99}},
        {"metadata": {"pair_id": 42}},
        {"metadata": {"pair_id": 7}},
    ]
    assert rank_at_k(results, target_pair_id=42, k=2) is True
    assert rank_at_k(results, target_pair_id=42, k=1) is False


def test_rank_at_k_miss():
    from retrieval_pipeline import rank_at_k
    results = [{"metadata": {"pair_id": i}} for i in range(20)]
    assert rank_at_k(results, target_pair_id=999, k=20) is False


def test_generate_report_contains_required_sections():
    from retrieval_pipeline import generate_report
    fake_results = [
        {
            "query_num": 1,
            "category": "short sleeve top",
            "archetype": "Casual chic",
            "hits": {1: True, 5: True, 10: True, 20: True},
            "top_score": 0.92,
            "encode_time": 0.5,
            "rembg_used": True,
        },
        {
            "query_num": 2,
            "category": "trousers",
            "archetype": "Professionnel moderne",
            "hits": {1: False, 5: True, 10: True, 20: True},
            "top_score": 0.71,
            "encode_time": 0.4,
            "rembg_used": False,
        },
    ]
    report = generate_report(fake_results, index_size=32225)
    assert "## Summary" in report
    assert "| Query |" in report
    assert "Rank-1" in report
    assert "Mean" in report
    assert "32,225" in report


def test_generate_report_mean_accuracy():
    from retrieval_pipeline import generate_report
    fake_results = [
        {
            "query_num": 1,
            "category": "tops",
            "archetype": "Casual chic",
            "hits": {1: True, 5: True, 10: True, 20: True},
            "top_score": 0.9,
            "encode_time": 0.5,
            "rembg_used": True,
        },
        {
            "query_num": 2,
            "category": "bottoms",
            "archetype": "Streetwear",
            "hits": {1: False, 5: False, 10: False, 20: False},
            "top_score": 0.5,
            "encode_time": 0.4,
            "rembg_used": False,
        },
    ]
    report = generate_report(fake_results, index_size=100)
    assert "50%" in report