"""
Catalogue indexer — run once to build DINOv3 embeddings for all catalogue items.

Usage:
    uv run python catalogue.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModel.from_pretrained(MODEL_ID)
MODEL.eval()


def embed_image(img: Image.Image) -> np.ndarray:
    """Return the 384-dim DINOv3 [CLS] embedding for a PIL image."""
    inputs = PROCESSOR(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = MODEL(**inputs)
    # [CLS] token is index 0 of last_hidden_state
    cls_vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls_vec[0].astype(np.float32)  # shape (384,)


def build_index(
    dataset_dir: Path,
    metadata: dict,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Embed every image in dataset_dir that appears in metadata.
    Saves embeddings.npy (N×384) and index_metadata.json (list of metadata dicts).
    Returns (embeddings_path, metadata_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = []
    index_meta = []

    image_files = sorted(dataset_dir.rglob("*.jpg"))
    total = len(image_files)

    for i, img_path in enumerate(image_files):
        filename = img_path.name
        if filename not in metadata:
            continue
        print(f"[{i+1}/{total}] Embedding {filename}", flush=True)
        img = Image.open(img_path).convert("RGB")
        vec = embed_image(img)
        embeddings.append(vec)
        index_meta.append(metadata[filename])

    if not embeddings:
        raise ValueError(f"No matching images found in {dataset_dir}. Check dataset_dir and metadata keys.")

    emb_array = np.stack(embeddings, axis=0).astype(np.float32)

    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "index_metadata.json"

    np.save(emb_path, emb_array)
    with open(meta_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"Saved {emb_array.shape[0]} embeddings to {emb_path}")
    return emb_path, meta_path


if __name__ == "__main__":
    dataset_dir = Path("dataset/p13")
    metadata_path = dataset_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    build_index(
        dataset_dir=dataset_dir,
        metadata=metadata,
        out_dir=Path("data"),
    )
