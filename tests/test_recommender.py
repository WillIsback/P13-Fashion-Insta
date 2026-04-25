import json
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


FAKE_META = [
    {"fichier": "a.jpg", "categorie": "chemises", "articleType": "Shirt",
     "productDisplayName": "Red Shirt", "baseColour": "Red", "gender": "Men", "season": "Summer"},
    {"fichier": "b.jpg", "categorie": "t-shirts", "articleType": "Tshirts",
     "productDisplayName": "Blue Tshirt", "baseColour": "Blue", "gender": "Women", "season": "Fall"},
    {"fichier": "c.jpg", "categorie": "pantalons", "articleType": "Jeans",
     "productDisplayName": "Black Jeans", "baseColour": "Black", "gender": "Men", "season": "Fall"},
]

# Three embeddings: first two are close to query, third is far
FAKE_EMBEDDINGS = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

FAKE_QUERY_VEC = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_search_returns_top_n_sorted_by_similarity(tmp_path):
    from recommender import search

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    with open(meta_path, "w") as f:
        json.dump(FAKE_META, f)

    results = search(FAKE_QUERY_VEC, emb_path, meta_path, top_n=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["metadata"]["fichier"] == "a.jpg"


def test_search_result_structure(tmp_path):
    from recommender import search

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    with open(meta_path, "w") as f:
        json.dump(FAKE_META, f)

    results = search(FAKE_QUERY_VEC, emb_path, meta_path, top_n=3)

    for r in results:
        assert "score" in r
        assert "metadata" in r
        assert isinstance(r["score"], float)


def test_recommend_returns_correct_top_n(tmp_path):
    from recommender import recommend

    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "index_metadata.json"
    np.save(emb_path, FAKE_EMBEDDINGS)
    with open(meta_path, "w") as f:
        json.dump(FAKE_META, f)

    fake_img = Image.new("RGB", (64, 64))
    fake_vec = FAKE_QUERY_VEC

    with patch("recommender.embed_query", return_value=fake_vec):
        results = recommend(fake_img, emb_path, meta_path, top_n=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
