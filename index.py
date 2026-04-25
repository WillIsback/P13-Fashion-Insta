"""Rebuild DINOv3 embeddings index from data/catalogue.json.

Supports checkpoint resume — safe to interrupt and restart.

Usage:
    uv run python index.py
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from catalogue import embed_image

CATALOGUE_PATH = Path("data/catalogue.json")
OUT_DIR = Path("data")
CHECKPOINT_EMB = OUT_DIR / "index_checkpoint_embeddings.npy"
CHECKPOINT_META = OUT_DIR / "index_checkpoint_meta.json"
CHECKPOINT_STATE = OUT_DIR / "index_checkpoint.json"
BATCH_SIZE = 500


def load_checkpoint() -> tuple[list, list, int]:
    """Load partial progress. Returns (embeddings_list, meta_list, n_processed)."""
    if not CHECKPOINT_STATE.exists():
        return [], [], 0
    state = json.loads(CHECKPOINT_STATE.read_text())
    n = state["processed"]
    embeddings = list(np.load(CHECKPOINT_EMB))
    meta = json.loads(CHECKPOINT_META.read_text())
    print(f"  Resuming from checkpoint: {n} items already embedded")
    return embeddings, meta, n


def save_checkpoint(embeddings: list, meta: list) -> None:
    """Save partial progress to checkpoint files."""
    np.save(CHECKPOINT_EMB, np.stack(embeddings))
    CHECKPOINT_META.write_text(json.dumps(meta))
    CHECKPOINT_STATE.write_text(json.dumps({"processed": len(embeddings)}))


def clear_checkpoint() -> None:
    """Delete checkpoint files after successful completion."""
    for f in [CHECKPOINT_EMB, CHECKPOINT_META, CHECKPOINT_STATE]:
        f.unlink(missing_ok=True)


def build_index() -> None:
    with open(CATALOGUE_PATH) as f:
        catalogue = json.load(f)

    embeddings, meta, n_processed = load_checkpoint()
    remaining = catalogue[n_processed:]

    print(f"  {len(catalogue):,} catalogue items, {n_processed:,} already done, "
          f"{len(remaining):,} to process")

    pbar = tqdm(remaining, desc="Embedding", unit="img")
    for i, item in enumerate(pbar):
        img_path = Path(item["path"])
        if not img_path.exists():
            print(f"\n[warn] missing: {img_path}, skipping")
            continue
        img = Image.open(img_path).convert("RGB")
        vec = embed_image(img)
        embeddings.append(vec)
        meta.append(item)

        if (i + 1) % BATCH_SIZE == 0:
            save_checkpoint(embeddings, meta)
            pbar.set_postfix({"checkpoint": f"{len(embeddings):,} saved"})

    if not embeddings:
        raise ValueError("No embeddings produced. Check data/catalogue.json paths.")

    emb_array = np.stack(embeddings).astype(np.float32)
    np.save(OUT_DIR / "embeddings.npy", emb_array)
    with open(OUT_DIR / "index_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    clear_checkpoint()
    print(f"\nIndex built: {emb_array.shape[0]:,} embeddings → data/embeddings.npy")
    print(f"Metadata    → data/index_metadata.json")


if __name__ == "__main__":
    build_index()