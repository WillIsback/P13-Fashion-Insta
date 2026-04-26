"""
Build full-catalogue embeddings for CLIP-based models (FashionCLIP, Marqo FashionSigLIP).

Produces:
    data/embeddings_fashion_clip.npy
    data/embeddings_marqo_fashion_siglip.npy

Both share the same item order as data/index_metadata.json.
Supports checkpoint resume — safe to interrupt and restart.

Usage:
    uv run python -m backend.scripts.build_clip_indices [--model fashion_clip|marqo_fashion_siglip]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.scripts.generate_retrieval_report import (
    MODELS,
    embed_image,
    load_model,
    _purge_cuda,
)

METADATA_PATH = Path("data/index_metadata.json")
OUT_DIR = Path("data")
BATCH_SIZE = 200

CLIP_MODELS = [k for k in MODELS if k != "dinov3_vith16"]


def checkpoint_paths(model_key: str) -> tuple[Path, Path]:
    return (
        OUT_DIR / f"ckpt_{model_key}_emb.npy",
        OUT_DIR / f"ckpt_{model_key}_state.json",
    )


def load_checkpoint(model_key: str) -> tuple[list, int]:
    emb_path, state_path = checkpoint_paths(model_key)
    if not state_path.exists():
        return [], 0
    state = json.loads(state_path.read_text())
    n = state["processed"]
    embeddings = list(np.load(emb_path))
    print(f"  Checkpoint: {n:,} items already embedded")
    return embeddings, n


def save_checkpoint(model_key: str, embeddings: list) -> None:
    emb_path, state_path = checkpoint_paths(model_key)
    np.save(emb_path, np.stack(embeddings))
    state_path.write_text(json.dumps({"processed": len(embeddings)}))


def clear_checkpoint(model_key: str) -> None:
    for p in checkpoint_paths(model_key):
        p.unlink(missing_ok=True)


def build_index(model_key: str) -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    out_file = OUT_DIR / f"embeddings_{model_key}.npy"

    if out_file.exists():
        print(f"  {out_file} already exists — delete it to rebuild.")
        return

    print(f"\nLoading model {MODELS[model_key]['label']}...")
    model, processor, device = load_model(model_key)

    embeddings, n_done = load_checkpoint(model_key)
    remaining = metadata[n_done:]
    print(f"  {len(metadata):,} catalogue items — {n_done:,} done, {len(remaining):,} to embed")

    pbar = tqdm(remaining, desc=f"  {model_key}", unit="img")
    for i, item in enumerate(pbar):
        img_path = Path(item["path"])
        if not img_path.exists():
            # Insert zero vector to preserve alignment with metadata
            dim = MODELS[model_key]["embed_dim"]
            embeddings.append(np.zeros(dim, dtype=np.float32))
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            vec = embed_image(model_key, model, processor, device, img)
            embeddings.append(vec)
        except Exception as e:
            print(f"\n  [warn] {img_path}: {e} — inserting zero vector")
            embeddings.append(np.zeros(MODELS[model_key]["embed_dim"], dtype=np.float32))

        if (i + 1) % BATCH_SIZE == 0:
            save_checkpoint(model_key, embeddings)
            pbar.set_postfix({"saved": len(embeddings)})

    del model, processor
    _purge_cuda()

    emb_array = np.stack(embeddings).astype(np.float32)
    np.save(out_file, emb_array)
    clear_checkpoint(model_key)
    print(f"\n  Index saved → {out_file}  ({emb_array.shape[0]:,} × {emb_array.shape[1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=CLIP_MODELS + ["all"],
        default="all",
        help="Which model index to build (default: all)",
    )
    args = parser.parse_args()

    targets = CLIP_MODELS if args.model == "all" else [args.model]
    for model_key in targets:
        build_index(model_key)


if __name__ == "__main__":
    main()
