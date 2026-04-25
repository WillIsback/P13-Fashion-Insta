import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


def make_fake_model():
    """Returns a mock processor and model that produce 384-dim embeddings."""
    processor = MagicMock()
    processor.return_value = {"pixel_values": MagicMock()}

    model = MagicMock()
    # last_hidden_state[:, 0, :] is the [CLS] token → shape (1, 384)
    fake_output = MagicMock()
    fake_output.last_hidden_state = MagicMock()
    fake_output.last_hidden_state.__getitem__ = lambda self, idx: MagicMock(
        detach=lambda: MagicMock(
            cpu=lambda: MagicMock(
                numpy=lambda: np.zeros((1, 384), dtype=np.float32)
            )
        )
    )
    model.return_value = fake_output
    return processor, model


def test_embed_image_returns_384_vector(tmp_path):
    from catalogue import embed_image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    processor, model = make_fake_model()

    with patch("catalogue.PROCESSOR", processor), patch("catalogue.MODEL", model):
        vec = embed_image(img)

    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_build_index_creates_files(tmp_path):
    from catalogue import build_index

    # Create a tiny fake dataset: 2 images
    cat_dir = tmp_path / "dataset"
    cat_dir.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        Image.new("RGB", (64, 64)).save(cat_dir / name)

    meta = {
        "a.jpg": {"fichier": "a.jpg", "categorie": "test", "articleType": "Shirt",
                  "productDisplayName": "Test A", "baseColour": "Red",
                  "gender": "Men", "season": "Summer"},
        "b.jpg": {"fichier": "b.jpg", "categorie": "test", "articleType": "Shirt",
                  "productDisplayName": "Test B", "baseColour": "Blue",
                  "gender": "Women", "season": "Fall"},
    }

    processor, model = make_fake_model()

    with patch("catalogue.PROCESSOR", processor), patch("catalogue.MODEL", model):
        emb_path, meta_path = build_index(
            dataset_dir=cat_dir,
            metadata=meta,
            out_dir=tmp_path,
        )

    embeddings = np.load(emb_path)
    assert embeddings.shape == (2, 384)

    with open(meta_path) as f:
        saved_meta = json.load(f)
    assert len(saved_meta) == 2
    assert saved_meta[0]["fichier"] in ("a.jpg", "b.jpg")
