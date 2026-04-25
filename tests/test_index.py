import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch


def make_item(i, tmp_path):
    img_path = tmp_path / f"{i:06d}.jpg"
    Image.new("RGB", (64, 64)).save(img_path)
    return {
        "filename": f"{i:06d}.jpg",
        "path": str(img_path),
        "pair_id": i,
        "category_name": "short sleeve top",
        "category_id": 1,
        "fashn_category": "tops",
        "archetype": "Casual chic",
        "b_box": [],
        "split_source": "train",
    }


def fake_embed(img):
    return np.zeros(384, dtype=np.float32)


def test_load_checkpoint_returns_empty_when_no_file(tmp_path, monkeypatch):
    import index
    monkeypatch.setattr(index, "CHECKPOINT_STATE", tmp_path / "state.json")
    monkeypatch.setattr(index, "CHECKPOINT_EMB", tmp_path / "emb.npy")
    monkeypatch.setattr(index, "CHECKPOINT_META", tmp_path / "meta.json")

    embs, meta, n = index.load_checkpoint()
    assert embs == []
    assert meta == []
    assert n == 0


def test_save_and_load_checkpoint_roundtrip(tmp_path, monkeypatch):
    import index
    monkeypatch.setattr(index, "CHECKPOINT_STATE", tmp_path / "state.json")
    monkeypatch.setattr(index, "CHECKPOINT_EMB", tmp_path / "emb.npy")
    monkeypatch.setattr(index, "CHECKPOINT_META", tmp_path / "meta.json")

    fake_embs = [np.zeros(384, dtype=np.float32), np.ones(384, dtype=np.float32)]
    fake_meta = [{"pair_id": 1}, {"pair_id": 2}]
    index.save_checkpoint(fake_embs, fake_meta)

    embs, meta, n = index.load_checkpoint()
    assert n == 2
    assert len(embs) == 2
    assert meta == fake_meta


def test_build_index_produces_correct_output(tmp_path, monkeypatch):
    import index
    items = [make_item(i, tmp_path) for i in range(3)]
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(items))

    monkeypatch.setattr(index, "CATALOGUE_PATH", catalogue_path)
    monkeypatch.setattr(index, "OUT_DIR", tmp_path)
    monkeypatch.setattr(index, "CHECKPOINT_STATE", tmp_path / "state.json")
    monkeypatch.setattr(index, "CHECKPOINT_EMB", tmp_path / "emb.npy")
    monkeypatch.setattr(index, "CHECKPOINT_META", tmp_path / "meta.json")

    with patch("index.embed_image", side_effect=fake_embed):
        index.build_index()

    embs = np.load(tmp_path / "embeddings.npy")
    assert embs.shape == (3, 384)

    meta = json.loads((tmp_path / "index_metadata.json").read_text())
    assert len(meta) == 3
    assert meta[0]["pair_id"] == 0


def test_build_index_skips_missing_images(tmp_path, monkeypatch):
    import index
    items = [make_item(0, tmp_path), make_item(1, tmp_path)]
    items[1]["path"] = str(tmp_path / "nonexistent.jpg")
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(items))

    monkeypatch.setattr(index, "CATALOGUE_PATH", catalogue_path)
    monkeypatch.setattr(index, "OUT_DIR", tmp_path)
    monkeypatch.setattr(index, "CHECKPOINT_STATE", tmp_path / "state.json")
    monkeypatch.setattr(index, "CHECKPOINT_EMB", tmp_path / "emb.npy")
    monkeypatch.setattr(index, "CHECKPOINT_META", tmp_path / "meta.json")

    with patch("index.embed_image", side_effect=fake_embed):
        index.build_index()

    embs = np.load(tmp_path / "embeddings.npy")
    assert embs.shape == (1, 384)


def test_checkpoint_deleted_after_success(tmp_path, monkeypatch):
    import index
    items = [make_item(0, tmp_path)]
    catalogue_path = tmp_path / "catalogue.json"
    catalogue_path.write_text(json.dumps(items))

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(index, "CATALOGUE_PATH", catalogue_path)
    monkeypatch.setattr(index, "OUT_DIR", tmp_path)
    monkeypatch.setattr(index, "CHECKPOINT_STATE", state_path)
    monkeypatch.setattr(index, "CHECKPOINT_EMB", tmp_path / "emb.npy")
    monkeypatch.setattr(index, "CHECKPOINT_META", tmp_path / "meta.json")

    with patch("index.embed_image", side_effect=fake_embed):
        index.build_index()

    assert not state_path.exists()