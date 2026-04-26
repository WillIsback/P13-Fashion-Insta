import numpy as np
import pytest
import warnings
from PIL import Image
from unittest.mock import patch, MagicMock


EMBED_DIM = 768


def make_fake_marqo():
    """Returns mock (model, preprocess, tokenizer) producing 768-dim embeddings."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    vec[0] = 1.0  # non-zero so normalisation is meaningful

    model = MagicMock()
    model.encode_image.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: vec.reshape(1, -1))
    )
    model.encode_text.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: vec.reshape(1, -1))
    )
    model.parameters.return_value = iter([MagicMock(device="cpu")])

    preprocess = MagicMock(return_value=MagicMock(unsqueeze=lambda _: MagicMock()))
    tokenizer = MagicMock(return_value=MagicMock())

    return model, preprocess, tokenizer


def test_embed_image_returns_768_vector():
    from backend.core.embedder import embed_image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    fake = make_fake_marqo()

    with patch("backend.core.embedder._load_marqo", return_value=fake):
        vec = embed_image(img)

    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32


def test_embed_image_is_l2_normalised():
    from backend.core.embedder import embed_image

    img = Image.new("RGB", (224, 224))
    fake = make_fake_marqo()

    with patch("backend.core.embedder._load_marqo", return_value=fake):
        vec = embed_image(img)

    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_text_returns_768_vector():
    from backend.core.embedder import embed_text

    fake = make_fake_marqo()

    with patch("backend.core.embedder._load_marqo", return_value=fake):
        vec = embed_text("blue denim jeans")

    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32


# ── Legacy ─────────────────────────────────────────────────────────────────

def test_embed_image_dinov3_emits_deprecation_warning():
    from backend.core.embedder import embed_image_dinov3

    img = Image.new("RGB", (224, 224))

    processor = MagicMock(return_value={"pixel_values": MagicMock()})
    hidden = np.zeros((1, 1, 1280), dtype=np.float32)
    hidden[0, 0, 0] = 1.0
    fake_output = MagicMock()
    fake_output.last_hidden_state.__getitem__ = lambda self, idx: MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: hidden[:, 0, :])
    )
    model = MagicMock(return_value=fake_output)

    with patch("backend.core.embedder._load_dinov3", return_value=(processor, model)):
        with pytest.warns(DeprecationWarning, match="embed_image_dinov3"):
            vec = embed_image_dinov3(img)

    assert vec.shape == (1280,)
