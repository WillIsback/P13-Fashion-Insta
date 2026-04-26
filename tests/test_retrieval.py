import json
import numpy as np
import pytest
import warnings
from pathlib import Path
from PIL import Image
from unittest.mock import patch

FAKE_META = [
    {"filename": "a.jpg", "category_name": "Tees_Tanks", "fashn_category": "tops", "archetype": "Casual chic"},
    {"filename": "b.jpg", "category_name": "Pants", "fashn_category": "bottoms", "archetype": "Professionnel moderne"},
    {"filename": "c.jpg", "category_name": "Dresses", "fashn_category": "one-pieces", "archetype": "Bohème féminine"},
]

# Three embeddings: first two are close to query, third is far
FAKE_EMBEDDINGS = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

FAKE_QUERY_VEC = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_search_returns_top_n_sorted_by_similarity(tmp_path):
    from backend.core.retrieval import search

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    meta_path.write_text(json.dumps(FAKE_META))

    results = search(FAKE_QUERY_VEC, emb_path, meta_path, top_n=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["metadata"]["filename"] == "a.jpg"


def test_search_result_structure(tmp_path):
    from backend.core.retrieval import search

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    meta_path.write_text(json.dumps(FAKE_META))

    results = search(FAKE_QUERY_VEC, emb_path, meta_path, top_n=3)

    for r in results:
        assert "score" in r
        assert "metadata" in r
        assert isinstance(r["score"], float)


def test_recommend_returns_correct_top_n(tmp_path):
    from backend.core.retrieval import recommend

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    meta_path.write_text(json.dumps(FAKE_META))

    fake_img = Image.new("RGB", (64, 64))

    with patch("backend.core.retrieval.embed_query", return_value=FAKE_QUERY_VEC):
        results = recommend(fake_img, emb_path, meta_path, top_n=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_recommend_from_text_returns_correct_top_n(tmp_path):
    from backend.core.retrieval import recommend_from_text

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    meta_path.write_text(json.dumps(FAKE_META))

    with patch("backend.core.retrieval.embed_text", return_value=FAKE_QUERY_VEC):
        results = recommend_from_text("blue jeans", emb_path, meta_path, top_n=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


# ── Legacy ─────────────────────────────────────────────────────────────────

def test_embed_query_dinov3_emits_deprecation_warning():
    from backend.core.retrieval import embed_query_dinov3

    fake_img = Image.new("RGB", (64, 64))
    fake_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with patch("backend.core.embedder.embed_image_dinov3", return_value=fake_vec):
        with patch("backend.core.retrieval.remove", return_value=fake_img):
            with pytest.warns(DeprecationWarning, match="embed_query_dinov3"):
                vec = embed_query_dinov3(fake_img)

    assert vec is fake_vec


def test_recommend_dinov3_emits_deprecation_warning(tmp_path):
    from backend.core.retrieval import recommend_dinov3

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    meta_path.write_text(json.dumps(FAKE_META))

    fake_img = Image.new("RGB", (64, 64))

    with patch("backend.core.retrieval.embed_query_dinov3", return_value=FAKE_QUERY_VEC):
        with pytest.warns(DeprecationWarning, match="recommend_dinov3"):
            results = recommend_dinov3(fake_img, emb_path, meta_path, top_n=2)

    assert len(results) == 2
