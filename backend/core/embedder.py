"""
Marqo FashionSigLIP image and text embedder.

Returns 768-dim L2-normalised embeddings per image or text query.
Model is loaded lazily on first call and cached for the process lifetime.

Usage:
    from backend.core.embedder import embed_image, embed_text
    vec = embed_image(pil_image)   # np.ndarray shape (768,) float32
    vec = embed_text("blue jeans") # np.ndarray shape (768,) float32

Legacy:
    embed_image_dinov3() — DINOv3 ViT-H/16+ (1280-dim), kept for backwards compatibility.
"""
import warnings
from functools import lru_cache

import numpy as np
from PIL import Image
import torch
import open_clip

MODEL_ID = "Marqo/marqo-fashionSigLIP"
EMBED_DIM = 768

# ── Legacy constants (kept for any code that imports them) ─────────────────
_DINO_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
_DINO_EMBED_DIM = 1280


@lru_cache(maxsize=1)
def _load_marqo():
    """Load Marqo FashionSigLIP once; cache for the process lifetime."""
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{MODEL_ID}")
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{MODEL_ID}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, preprocess, tokenizer


def embed_image(img: Image.Image) -> np.ndarray:
    """
    Return the 768-dim Marqo FashionSigLIP embedding for a PIL image.
    Output is L2-normalised — safe for cosine similarity via dot product.
    """
    model, preprocess, _ = _load_marqo()
    device = next(model.parameters()).device
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(tensor, normalize=True).cpu().numpy()[0]
    return vec.astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """
    Return the 768-dim Marqo FashionSigLIP embedding for a text query.
    Output is L2-normalised — safe for cosine similarity via dot product.
    """
    model, _, tokenizer = _load_marqo()
    device = next(model.parameters()).device
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        vec = model.encode_text(tokens, normalize=True).cpu().numpy()[0]
    return vec.astype(np.float32)


# ── Legacy / Deprecated ────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_dinov3():
    """[DEPRECATED] Load DINOv3 ViT-H/16+ model."""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(_DINO_MODEL_ID)
    model = AutoModel.from_pretrained(_DINO_MODEL_ID)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return processor, model


# Alias kept so existing code that patches `_load_model` still works
_load_model = _load_dinov3


def embed_image_dinov3(img: Image.Image) -> np.ndarray:
    """
    [DEPRECATED] Use embed_image() instead (Marqo FashionSigLIP, 768-dim).

    Return the 1280-dim DINOv3 ViT-H/16+ [CLS] embedding for a PIL image.
    Output is L2-normalised.
    """
    warnings.warn(
        "embed_image_dinov3() is deprecated — use embed_image() (Marqo FashionSigLIP).",
        DeprecationWarning,
        stacklevel=2,
    )
    processor, model = _load_dinov3()
    inputs = processor(images=img, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
