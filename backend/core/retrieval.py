"""
Retrieval — embed a query and search the catalogue by cosine similarity.

Default embedder: Marqo FashionSigLIP (image + text, 768-dim).
Requires: data/embeddings_marqo_fashion_siglip.npy + data/index_metadata.json

Usage:
    from backend.core.retrieval import recommend, recommend_from_text
    results = recommend(pil_image, embeddings_path, metadata_path, top_n=5)
    results = recommend_from_text("blue jeans", embeddings_path, metadata_path, top_n=5)
    # results: [{"score": float, "metadata": dict}, ...]

Legacy:
    embed_query_dinov3() / recommend_dinov3() — DINOv3-based, kept for backwards compatibility.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.embedder import embed_image, embed_text


def embed_query(img: Image.Image) -> np.ndarray:
    """
    Prepare a user photo for retrieval:
    - Remove background with rembg (falls back to raw image on failure)
    - Embed with Marqo FashionSigLIP

    Returns 768-dim L2-normalised float32 vector.
    """
    try:
        img_no_bg = remove(img)
        img_rgb = img_no_bg.convert("RGB")
    except Exception as e:
        print(f"[warn] rembg failed ({e}), using raw image")
        img_rgb = img.convert("RGB")
    return embed_image(img_rgb)


def search(
    query_vec: np.ndarray,
    embeddings_path: Path,
    metadata_path: Path,
    top_n: int = 5,
) -> list[dict]:
    """
    Compute cosine similarity between query_vec and all catalogue embeddings.
    Returns top_n results sorted by descending score.
    Each result: {"score": float, "metadata": dict}
    """
    embeddings = np.load(embeddings_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    scores = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [
        {"score": float(scores[i]), "metadata": metadata[i]}
        for i in top_indices
    ]


def recommend(
    user_img: Image.Image,
    embeddings_path: Path,
    metadata_path: Path,
    top_n: int = 5,
) -> list[dict]:
    """Image-to-image pipeline: embed user photo → cosine search → top_n results."""
    query_vec = embed_query(user_img)
    return search(query_vec, embeddings_path, metadata_path, top_n=top_n)


def recommend_from_text(
    text_query: str,
    embeddings_path: Path,
    metadata_path: Path,
    top_n: int = 5,
) -> list[dict]:
    """Text-to-image pipeline: embed text query → cosine search → top_n results."""
    query_vec = embed_text(text_query)
    return search(query_vec, embeddings_path, metadata_path, top_n=top_n)


# ── Legacy / Deprecated ────────────────────────────────────────────────────


def embed_query_dinov3(img: Image.Image) -> np.ndarray:
    """
    [DEPRECATED] Use embed_query() instead (Marqo FashionSigLIP).
    DINOv3-based query embedding with background removal.
    """
    warnings.warn(
        "embed_query_dinov3() is deprecated — use embed_query() (Marqo FashionSigLIP).",
        DeprecationWarning,
        stacklevel=2,
    )
    from backend.core.embedder import embed_image_dinov3
    try:
        img_no_bg = remove(img)
        img_rgb = img_no_bg.convert("RGB")
    except Exception as e:
        print(f"[warn] rembg failed ({e}), using raw image")
        img_rgb = img.convert("RGB")
    return embed_image_dinov3(img_rgb)


def recommend_dinov3(
    user_img: Image.Image,
    embeddings_path: Path,
    metadata_path: Path,
    top_n: int = 5,
) -> list[dict]:
    """
    [DEPRECATED] Use recommend() instead (Marqo FashionSigLIP).
    DINOv3-based full pipeline.
    """
    warnings.warn(
        "recommend_dinov3() is deprecated — use recommend() (Marqo FashionSigLIP).",
        DeprecationWarning,
        stacklevel=2,
    )
    query_vec = embed_query_dinov3(img=user_img)
    return search(query_vec, embeddings_path, metadata_path, top_n=top_n)
