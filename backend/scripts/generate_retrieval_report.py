"""
Generate retrieval benchmark report comparing multiple embedders.

Runs image-to-image and text-to-image retrieval tests on the demo user photos
against the DeepFashion InShop catalogue, computing similarity scores,
then writes docs/Embedder/retrieval_comparison.md.

Usage:
    uv run python -m backend.scripts.generate_retrieval_report
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
HF_TOKEN = None
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().split("\n"):
        if line.startswith("HF_TOKEN="):
            HF_TOKEN = line.split("=", 1)[1].strip()
            print(f"[DEBUG] Loaded HF_TOKEN: {HF_TOKEN[:10]}...")
            break

if not HF_TOKEN:
    print("[WARNING] No HF_TOKEN found in .env file!")

EMBEDDINGS_PATH = Path("data/embeddings.npy")
METADATA_PATH = Path("data/index_metadata.json")
DEMO_DIR = Path("demo")
RESULTS_FILE = Path("docs/Embedder/retrieval_results.json")
REPORT_FILE = Path("docs/Embedder/retrieval_comparison.md")

MODELS = {
    "dinov3_vith16": {
        "label": "DINOv3 ViT-H/16+",
        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "embed_dim": 1280,
        "use_token": True,
    },
    "fashion_clip": {
        "label": "FashionCLIP ViT-B/32",
        "model_id": "patrickjohncyh/fashion-clip",
        "embed_dim": 512,
        "text_encoder": True,
    },
    "marqo_fashion_siglip": {
        "label": "Marqo FashionSigLIP",
        "model_id": "Marqo/marqo-fashionSigLIP",
        "embed_dim": 768,
        "text_encoder": True,
        "trust_remote_code": True,
    },
}

TEXT_QUERIES = [
    "red summer dress",
    "blue denim jeans",
    "black formal jacket",
    "white cotton t-shirt",
    "green casual sweater",
    "brown leather boots",
]

# Intra-catalogue sample size for similarity estimation
INTRA_SIM_SAMPLE = 200


def load_catalogue() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return embeddings, metadata


def load_model(model_key: str):
    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    info = MODELS[model_key]
    model_id = info["model_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Loading {model_key}: {model_id} → {device}")

    if "dinov3" in model_key:
        token = HF_TOKEN if (info.get("use_token") and HF_TOKEN) else None
        print(f"[DEBUG] DINOv3: using token = {token is not None}")
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token).to(device)
    elif "marqo" in model_key:
        print(f"[DEBUG] Marqo: trust_remote_code=True")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Use device_map at load time to avoid meta-tensor .cuda() crash
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device,
        )
    else:
        print(f"[DEBUG] Fashion-CLIP: CLIPProcessor/CLIPModel")
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)

    model.eval()
    return model, processor, device


def _normalise(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def embed_image(model_key: str, model, processor, device: str, img: Image.Image) -> np.ndarray:
    import torch
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if "dinov3" in model_key:
            out = model(**inputs)
            vec = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
        else:
            vec = model.get_image_features(**inputs).cpu().numpy()[0]
    return _normalise(vec.astype(np.float32))


def embed_text(model_key: str, model, processor, device: str, text: str) -> np.ndarray:
    import torch
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy()[0]
    return _normalise(vec.astype(np.float32))


def build_cat_embeddings(model_key: str, model, processor, device: str, metadata: list, max_items: int = 500) -> tuple[np.ndarray, list[int]]:
    """Embed a catalogue subset. Returns (embeddings, valid_meta_indices)."""
    vecs, indices = [], []
    for i, meta in enumerate(tqdm(metadata[:max_items], desc="  Embedding catalogue", leave=False)):
        img_path = Path(meta["path"])  # path already relative to repo root
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            vecs.append(embed_image(model_key, model, processor, device, img))
            indices.append(i)
        except Exception:
            pass
    return np.array(vecs, dtype=np.float32) if vecs else np.empty((0,)), indices


def compute_intra_sim(embeddings: np.ndarray) -> float:
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    sample = embeddings if n <= 500 else embeddings[np.random.choice(n, 500, replace=False)]
    sims = np.dot(sample, sample.T)
    mask = ~np.eye(sample.shape[0], dtype=bool)
    return float(np.mean(sims[mask]))


def run_image_retrieval(model_key: str, model, processor, device: str,
                        user_img: Image.Image, gallery_embeddings: np.ndarray,
                        metadata: list, top_k: int = 5) -> list[dict]:
    query = embed_image(model_key, model, processor, device, user_img)

    if model_key == "dinov3_vith16" and gallery_embeddings.shape[1] == 1280:
        sims = np.dot(gallery_embeddings, query)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {"rank": i + 1, "idx": int(idx), "score": float(sims[idx]),
             "category": metadata[idx].get("category_name", "unknown")}
            for i, idx in enumerate(top_idx)
        ]

    # CLIP/SigLIP: embed catalogue subset
    cat_embs, cat_meta_indices = build_cat_embeddings(model_key, model, processor, device, metadata)
    if len(cat_embs) == 0:
        return []
    sims = np.dot(cat_embs, query)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"rank": i + 1, "idx": cat_meta_indices[idx], "score": float(sims[idx]),
         "category": metadata[cat_meta_indices[idx]].get("category_name", "unknown")}
        for i, idx in enumerate(top_idx)
    ]


def run_text_retrieval(model_key: str, model, processor, device: str,
                       text_query: str, metadata: list, top_k: int = 5) -> list[dict]:
    query = embed_text(model_key, model, processor, device, text_query)
    cat_embs, cat_meta_indices = build_cat_embeddings(model_key, model, processor, device, metadata)
    if len(cat_embs) == 0:
        return []
    sims = np.dot(cat_embs, query)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"rank": i + 1, "idx": cat_meta_indices[idx], "score": float(sims[idx]),
         "category": metadata[cat_meta_indices[idx]].get("category_name", "unknown")}
        for i, idx in enumerate(top_idx)
    ]


# ── Report generator ───────────────────────────────────────────────────────────

def write_report(results: dict, metadata: list) -> None:
    from datetime import date

    model_keys = list(MODELS.keys())
    model_labels = {k: MODELS[k]["label"] for k in model_keys}
    demo_users = sorted(DEMO_DIR.glob("user_*.png"))
    user_names = [p.name for p in demo_users]

    lines = []

    def h(text): lines.append(text)
    def sep(): lines.append("")
    def rule(): lines.append("---"); sep()

    h(f"# Retrieval Benchmark — Model Comparison")
    sep()
    h(f"**Date:** {date.today()}  ")
    h(f"**Catalogue:** DeepFashion InShop ({len(metadata):,} items)  ")
    h(f"**Queries:** {len(demo_users)} demo user photos + {len(TEXT_QUERIES)} text queries  ")
    h(f"**Metric:** Top-1 cosine similarity  ")
    sep()
    rule()

    # ── Section 1: Models ──
    h("## Models")
    sep()
    h("| Key | Label | Embed dim | Text encoder |")
    h("|-----|-------|-----------|-------------|")
    for k, info in MODELS.items():
        te = "✓" if info.get("text_encoder") else "—"
        h(f"| `{k}` | {info['label']} | {info['embed_dim']} | {te} |")
    sep()
    rule()

    # ── Section 2: Intra-catalogue similarity ──
    h("## 1. Intra-Catalogue Similarity")
    sep()
    h("> Mean pairwise cosine similarity within a catalogue sample.")
    h("> **Lower = better separation.** ≥ 0.55 signals near-collapse.")
    sep()
    h("| Model | Mean sim (↓ better) | Interpretation |")
    h("|-------|---------------------|----------------|")
    intra = results.get("intra_catalogue_sim", {})
    for k in model_keys:
        v = intra.get(k)
        if v is None or not isinstance(v, (int, float)):
            h(f"| {model_labels[k]} | — | — |")
            continue
        if v < 0.35:
            interp = "Good separation"
        elif v < 0.50:
            interp = "Moderate separation"
        else:
            interp = "Near-collapse"
        h(f"| {model_labels[k]} | {v:.4f} | {interp} |")
    sep()
    rule()

    # ── Section 3: Image-to-image top-1 scores ──
    h("## 2. Image-to-Image Retrieval — Top-1 Scores")
    sep()
    h("> Cosine similarity between user photo embedding and best catalogue match.")
    sep()

    i2i = results.get("image_to_image", {})
    header = "| User | " + " | ".join(model_labels[k] for k in model_keys) + " |"
    divider = "|------|" + "|".join(["------"] * len(model_keys)) + "|"
    h(header)
    h(divider)

    col_scores = {k: [] for k in model_keys}
    for user_name in user_names:
        user_key = user_name.replace(".png", "")
        row = f"| {user_name} |"
        for k in model_keys:
            entry = i2i.get(k, {}).get(user_key, {})
            if "top1_score" in entry:
                s = entry["top1_score"]
                col_scores[k].append(s)
                row += f" {s:.3f} |"
            elif "error" in entry:
                row += " ERROR |"
            else:
                row += " — |"
        h(row)

    # Average row
    avg_row = "| **Average** |"
    for k in model_keys:
        vals = col_scores[k]
        avg_row += f" **{np.mean(vals):.3f}** |" if vals else " — |"
    h(avg_row)
    sep()
    rule()

    # ── Section 4: Retrieved categories ──
    h("## 3. Retrieved Categories (Image-to-Image, Top-1)")
    sep()
    h("| User | " + " | ".join(model_labels[k] for k in model_keys) + " |")
    h("|------|" + "|".join(["------"] * len(model_keys)) + "|")
    for user_name in user_names:
        user_key = user_name.replace(".png", "")
        row = f"| {user_name} |"
        for k in model_keys:
            entry = i2i.get(k, {}).get(user_key, {})
            top5 = entry.get("top5", [])
            cat = top5[0].get("category", "—") if top5 else "—"
            row += f" {cat} |"
        h(row)
    sep()
    rule()

    # ── Section 5: Text-to-image ──
    text_model_keys = [k for k in model_keys if MODELS[k].get("text_encoder")]
    if text_model_keys:
        h("## 4. Text-to-Image Retrieval — Top-1 Scores")
        sep()
        h("> Cosine similarity between text query embedding and best catalogue image match.")
        sep()
        h("| Query | " + " | ".join(model_labels[k] for k in text_model_keys) + " |")
        h("|-------|" + "|".join(["------"] * len(text_model_keys)) + "|")
        t2i = results.get("text_to_image", {})
        for query in TEXT_QUERIES:
            row = f"| {query} |"
            for k in text_model_keys:
                entry = t2i.get(k, {}).get(query, {})
                if "top1_score" in entry:
                    row += f" {entry['top1_score']:.3f} ({entry.get('top1_category', '?')}) |"
                elif "error" in entry:
                    row += " ERROR |"
                else:
                    row += " — |"
            h(row)
        sep()
        rule()

    # ── Section 6: Visual comparison ──
    h("## 5. Visual Comparison per User (Top-3 matches, DINOv3 ViT-H)")
    sep()
    h("> Query photo and top-3 retrieved catalogue items using DINOv3 ViT-H (production embedder).")
    sep()

    for user_name in user_names:
        user_key = user_name.replace(".png", "")
        h(f"### {user_name}")
        sep()

        # User photo
        h(f"**Query:** `demo/{user_name}`")
        sep()

        top5 = i2i.get("dinov3_vith16", {}).get(user_key, {}).get("top5", [])
        if top5:
            h("| Rank | Score | Category | Path |")
            h("|------|-------|----------|------|")
            for r in top5[:3]:
                idx = r["idx"]
                cat = metadata[idx].get("category_name", "?") if idx < len(metadata) else "?"
                path = metadata[idx].get("path", "?") if idx < len(metadata) else "?"
                h(f"| #{r['rank']} | {r['score']:.3f} | {cat} | `{path}` |")
        else:
            h("_No results_")
        sep()

    REPORT_FILE.write_text("\n".join(lines))
    print(f"\n  Report written → {REPORT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading catalogue embeddings...")
    gallery_embeddings, metadata = load_catalogue()
    print(f"  Catalogue: {gallery_embeddings.shape[0]:,} items, {gallery_embeddings.shape[1]} dims")

    demo_users = sorted(DEMO_DIR.glob("user_*.png"))
    print(f"  Demo users: {len(demo_users)}")

    results = {"image_to_image": {}, "text_to_image": {}, "intra_catalogue_sim": {}}

    # ── Image-to-image ──────────────────────────────────────────────────────
    print("\n=== IMAGE-TO-IMAGE RETRIEVAL ===")
    for model_key in tqdm(MODELS.keys(), desc="Models"):
        print(f"\n{model_key}:")
        results["image_to_image"][model_key] = {}

        try:
            model, processor, device = load_model(model_key)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            continue

        for user_path in tqdm(demo_users, leave=False):
            user_num = user_path.stem.split("_")[1]
            user_key = f"user_{user_num}"
            user_img = Image.open(user_path).convert("RGB")
            try:
                ret = run_image_retrieval(model_key, model, processor, device,
                                          user_img, gallery_embeddings, metadata)
                if ret:
                    results["image_to_image"][model_key][user_key] = {
                        "top1_score": ret[0]["score"],
                        "top5": ret,
                    }
                    print(f"  user_{user_num}: {ret[0]['score']:.3f} ({ret[0]['category']})")
                else:
                    results["image_to_image"][model_key][user_key] = {"error": "empty"}
                    print(f"  user_{user_num}: no results")
            except Exception as e:
                print(f"  user_{user_num}: ERROR — {e}")
                results["image_to_image"][model_key][user_key] = {"error": str(e)}

        # Intra-catalogue sim on embedded sample
        print(f"  Computing intra-catalogue similarity…")
        try:
            if model_key == "dinov3_vith16":
                intra = compute_intra_sim(gallery_embeddings)
            else:
                cat_embs, _ = build_cat_embeddings(model_key, model, processor, device,
                                                   metadata, max_items=INTRA_SIM_SAMPLE)
                intra = compute_intra_sim(cat_embs) if len(cat_embs) > 1 else float("nan")
            results["intra_catalogue_sim"][model_key] = intra
            print(f"  Intra-catalogue sim: {intra:.4f}")
        except Exception as e:
            print(f"  Intra-sim ERROR: {e}")

        del model, processor  # free VRAM before next model

    # ── Text-to-image ───────────────────────────────────────────────────────
    print("\n=== TEXT-TO-IMAGE RETRIEVAL ===")
    text_model_keys = [k for k in MODELS if MODELS[k].get("text_encoder")]
    for model_key in tqdm(text_model_keys, desc="Text models"):
        print(f"\n{model_key}:")
        results["text_to_image"][model_key] = {}

        try:
            model, processor, device = load_model(model_key)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            continue

        for text_query in tqdm(TEXT_QUERIES, leave=False):
            try:
                ret = run_text_retrieval(model_key, model, processor, device, text_query, metadata)
                if ret:
                    results["text_to_image"][model_key][text_query] = {
                        "top1_score": ret[0]["score"],
                        "top1_category": ret[0]["category"],
                        "top5": ret,
                    }
                    print(f"  '{text_query}': {ret[0]['score']:.3f} ({ret[0]['category']})")
                else:
                    results["text_to_image"][model_key][text_query] = {"error": "empty"}
                    print(f"  '{text_query}': no results")
            except Exception as e:
                print(f"  '{text_query}': ERROR — {e}")
                results["text_to_image"][model_key][text_query] = {"error": str(e)}

        del model, processor

    # ── Save JSON + MD ──────────────────────────────────────────────────────
    print("\n=== SAVING RESULTS ===")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"  JSON → {RESULTS_FILE}")

    write_report(results, metadata)


if __name__ == "__main__":
    main()
