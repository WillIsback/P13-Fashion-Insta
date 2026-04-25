"""
Recommender — rembg background removal + DINOv3 embedding + cosine similarity search.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove
from sklearn.metrics.pairwise import cosine_similarity

from catalogue import embed_image


def embed_query(img: Image.Image) -> np.ndarray:
    """
    Remove background from user photo, then embed with DINOv3.
    Falls back to raw image if rembg fails.
    Returns 384-dim float32 vector.
    """
    try:
        img_no_bg = remove(img)
        # rembg returns RGBA — convert back to RGB for DINOv3
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
    Returns list of top_n dicts: {"score": float, "metadata": dict}
    sorted by descending score.
    """
    embeddings = np.load(embeddings_path)  # (N, D)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # cosine_similarity expects 2D arrays
    query_2d = query_vec.reshape(1, -1)
    scores = cosine_similarity(query_2d, embeddings)[0]  # (N,)

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
    """
    Full query pipeline: embed user image → cosine search → top_n results.
    Each result: {"score": float, "metadata": dict}
    """
    query_vec = embed_query(user_img)
    return search(query_vec, embeddings_path, metadata_path, top_n=top_n)
