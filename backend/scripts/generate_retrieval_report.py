"""
Generate retrieval benchmark report comparing multiple embedders.

Runs image-to-image and text-to-image retrieval tests on the demo user photos
against the DeepFashion InShop catalogue, then writes docs/Embedder/retrieval_comparison.md.

Usage:
    uv run python -m backend.scripts.generate_retrieval_report
"""
import json
import sys
from datetime import date
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
            break

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
        "text_encoder": False,
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
    "women's red summer dress",
    "men's blue denim jeans",
    "men's black formal jacket",
    "men's white cotton t-shirt",
    "women's green casual sweater",
    "women's brown leather boots",
    "women's floral blouse",
    "men's chino trousers",
]

# Labels used to annotate retrieved catalogue images (gender + garment type).
# Only used by annotate_hits() — separate from TEXT_QUERIES benchmark.
ANNOTATION_LABELS = [
    "men's jacket", "men's vest", "men's coat",
    "men's jeans", "men's denim pants",
    "men's dress shirt", "men's polo shirt", "men's casual shirt",
    "men's t-shirt", "men's tank top",
    "men's trousers", "men's chinos", "men's shorts",
    "men's sweater", "men's hoodie",
    "women's jacket", "women's coat", "women's blazer",
    "women's jeans", "women's denim shorts",
    "women's blouse", "women's shirt",
    "women's t-shirt", "women's top", "women's tank top",
    "women's trousers", "women's shorts",
    "women's dress", "women's skirt",
    "women's sweater", "women's cardigan",
    "shoes", "boots", "sneakers", "sandals",
]

# Paths to full-catalogue embeddings for CLIP models (built by build_clip_indices.py)
CLIP_INDEX = {k: Path(f"data/embeddings_{k}.npy") for k in MODELS if k != "dinov3_vith16"}


# ── Model loading ──────────────────────────────────────────────────────────────

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

    if "dinov3" in model_key:
        token = HF_TOKEN if (info.get("use_token") and HF_TOKEN) else None
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token).to(device)
    elif "marqo" in model_key:
        import open_clip
        model, _, preprocess_val = open_clip.create_model_and_transforms(f"hf-hub:{model_id}")
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
        model = model.to(device)
        processor = {"preprocess": preprocess_val, "tokenizer": tokenizer}
    else:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)

    model.eval()
    return model, processor, device


def _purge_cuda():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _normalise(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def embed_image(model_key: str, model, processor, device, img: Image.Image) -> np.ndarray:
    import torch
    with torch.no_grad():
        if "dinov3" in model_key:
            inputs = {k: v.to(device) for k, v in processor(images=img, return_tensors="pt").items()}
            vec = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]
            return _normalise(vec.astype(np.float32))
        elif "marqo" in model_key:
            # OpenCLIP API: preprocess → tensor, encode_image with normalize=True
            img_tensor = processor["preprocess"](img).unsqueeze(0).to(device)
            vec = model.encode_image(img_tensor, normalize=True).cpu().numpy()[0]
            return vec.astype(np.float32)
        else:
            inputs = {k: v.to(device) for k, v in processor(images=img, return_tensors="pt").items()}
            out = model.get_image_features(**inputs)
            vec = out.cpu().numpy()[0] if hasattr(out, "cpu") else (
                out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output
            ).cpu().numpy()[0]
            return _normalise(vec.astype(np.float32))


def embed_text(model_key: str, model, processor, device, text: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        if "marqo" in model_key:
            # OpenCLIP API: tokenizer returns tensor directly, encode_text with normalize=True
            tokens = processor["tokenizer"]([text]).to(device)
            vec = model.encode_text(tokens, normalize=True).cpu().numpy()[0]
            return vec.astype(np.float32)
        else:
            inputs = {k: v.to(device) for k, v in processor(text=text, return_tensors="pt", padding=True).items()}
            out = model.get_text_features(**inputs)
            vec = out.cpu().numpy()[0] if hasattr(out, "cpu") else (
                out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output
            ).cpu().numpy()[0]
            return _normalise(vec.astype(np.float32))



def top_k_results(sims: np.ndarray, meta_indices: list[int], metadata: list, top_k: int = 5) -> list[dict]:
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "rank": i + 1,
            "idx": meta_indices[idx],
            "cat_emb_pos": int(idx),          # position in cat_embs, used for text annotation
            "score": float(sims[idx]),
            "category": metadata[meta_indices[idx]].get("category_name", "unknown"),
        }
        for i, idx in enumerate(top_idx)
    ]


def annotate_hits(hits: list[dict], cat_embs: np.ndarray, annotation_embs: dict) -> None:
    """For each hit, find the closest ANNOTATION_LABEL in embedding space and store it in-place."""
    if not annotation_embs:
        return
    labels = list(annotation_embs.keys())
    label_matrix = np.array([annotation_embs[l] for l in labels], dtype=np.float32)  # (N_labels, dim)
    for h in hits:
        pos = h.get("cat_emb_pos")
        if pos is None or pos >= len(cat_embs):
            continue
        sims = label_matrix @ cat_embs[pos]   # (N_labels,)
        best = int(np.argmax(sims))
        h["clip_annotation"] = labels[best]
        h["clip_annotation_score"] = float(sims[best])


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_model(model_key: str, gallery_embeddings: np.ndarray, metadata: list,
              demo_users: list[Path]) -> dict:
    """Run all image and text queries for one model. Returns per-model result dict."""
    info = MODELS[model_key]
    result = {"image": {}, "text": {}}

    try:
        model, processor, device = load_model(model_key)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return result

    # Load full-catalogue embeddings
    if "dinov3" in model_key:
        cat_embs = gallery_embeddings
    else:
        index_path = CLIP_INDEX[model_key]
        if not index_path.exists():
            print(f"  Index manquant — lance d'abord :")
            print(f"    uv run python -m backend.scripts.build_clip_indices --model {model_key}")
            del model, processor
            _purge_cuda()
            return result
        cat_embs = np.load(index_path)

    cat_indices = list(range(len(metadata)))
    print(f"  Catalogue: {cat_embs.shape[0]:,} items × {cat_embs.shape[1]} dims")

    # Pre-compute embeddings for annotation labels and text queries (text-encoder models only)
    annotation_embs: dict[str, np.ndarray] = {}
    text_query_embs: dict[str, np.ndarray] = {}
    if info.get("text_encoder"):
        print("  Pre-computing annotation label embeddings...")
        for label in ANNOTATION_LABELS:
            annotation_embs[label] = embed_text(model_key, model, processor, device, label)
        print("  Pre-computing text query embeddings...")
        for qt in TEXT_QUERIES:
            text_query_embs[qt] = embed_text(model_key, model, processor, device, qt)

    # Image queries
    for user_path in tqdm(demo_users, desc="  Image queries", leave=False):
        user_key = user_path.stem
        try:
            query = embed_image(model_key, model, processor, device, Image.open(user_path).convert("RGB"))
            sims = np.dot(cat_embs, query)
            hits = top_k_results(sims, cat_indices, metadata)
            annotate_hits(hits, cat_embs, annotation_embs)
            result["image"][user_key] = {"top1_score": hits[0]["score"], "top5": hits}
            annotation = f" → '{hits[0].get('clip_annotation', '')}'" if annotation_embs else ""
            print(f"    {user_key}: {hits[0]['score']:.3f} ({hits[0]['category']}){annotation}")
        except Exception as e:
            result["image"][user_key] = {"error": str(e)}
            print(f"    {user_key}: ERROR — {e}")

    # Text queries (only for models with a text encoder)
    if info.get("text_encoder"):
        for query_text in tqdm(TEXT_QUERIES, desc="  Text queries", leave=False):
            try:
                query = text_query_embs[query_text]  # already computed above
                sims = np.dot(cat_embs, query)
                hits = top_k_results(sims, cat_indices, metadata)
                annotate_hits(hits, cat_embs, annotation_embs)
                result["text"][query_text] = {"top1_score": hits[0]["score"], "top1_category": hits[0]["category"], "top5": hits}
                print(f"    '{query_text}': {hits[0]['score']:.3f} ({hits[0]['category']})")
            except Exception as e:
                result["text"][query_text] = {"error": str(e)}
                print(f"    '{query_text}': ERROR — {e}")

    del model, processor
    _purge_cuda()
    return result


# ── Report ─────────────────────────────────────────────────────────────────────

EXAMPLES_DIR = REPORT_FILE.parent / "examples"
TOP_K_SHOW = 3     # how many retrieved images to show per cell


def _stage(src: str, name: str) -> str:
    """Copy src (relative to repo root) into EXAMPLES_DIR, return markdown-relative path."""
    import shutil
    EXAMPLES_DIR.mkdir(exist_ok=True)
    src_path = Path(src)
    dest = EXAMPLES_DIR / (name + src_path.suffix)
    if src_path.exists() and not dest.exists():
        shutil.copy2(src_path, dest)
    return f"examples/{dest.name}"


def _img(rel_path: str, alt: str = "") -> str:
    return f"![{alt}]({rel_path})"


def _hits_cell(hits: list[dict], metadata: list, cell_id: str) -> str:
    """Build a cell showing top-k retrieved images with score + CLIP annotation."""
    parts = []
    for h in hits[:TOP_K_SHOW]:
        idx = h.get("idx")
        score = h.get("score", 0)
        cat = h.get("category", "?")
        annotation = h.get("clip_annotation")
        if idx is not None and idx < len(metadata):
            src = metadata[idx].get("path", "")
            if src:
                staged = _stage(src, f"cat_{idx}")
                alt = f"{score:.3f} — {cat}" + (f", {annotation}" if annotation else "")
                caption = f"{score:.3f} — {cat}"
                if annotation:
                    caption += f"<br><sub><i>{annotation}</i></sub>"
                parts.append(_img(staged, alt) + f"<br><sub>{caption}</sub>")
            else:
                parts.append(f"idx {idx}")
        else:
            parts.append("—")
    return " ".join(parts) if parts else "—"


def write_report(results: dict, metadata: list, demo_users: list[Path]) -> None:
    model_keys = list(MODELS.keys())
    text_model_keys = [k for k in model_keys if MODELS[k].get("text_encoder")]
    lines = []

    lines.append("# Rapport de benchmark — Retrieval")
    lines.append("")

    # ── Tableau 1 : Conditions ─────────────────────────────────────────────
    lines.append("## Conditions du benchmark")
    lines.append("")
    lines.append("| Paramètre | Valeur |")
    lines.append("|-----------|--------|")
    lines.append(f"| Date | {date.today()} |")
    lines.append(f"| Catalogue | DeepFashion InShop — {len(metadata):,} articles |")
    lines.append(f"| Données client (requêtes image) | {len(demo_users)} photos utilisateurs |")
    lines.append(f"| Requêtes texte | {len(TEXT_QUERIES)} requêtes |")
    lines.append(f"| Métrique | Similarité cosinus Top-{TOP_K_SHOW} |")
    lines.append(f"| Catalogue CLIP | index complet ({len(metadata):,} articles) |")
    lines.append("")
    lines.append("**Modèles évalués :**")
    lines.append("")
    lines.append("| Modèle | Dimension | Encodeur texte |")
    lines.append("|--------|-----------|----------------|")
    for k, info in MODELS.items():
        te = "Oui" if info.get("text_encoder") else "Non"
        lines.append(f"| {info['label']} | {info['embed_dim']} | {te} |")
    lines.append("")

    # ── Tableau 2a : Requêtes image ────────────────────────────────────────
    lines.append("## Résultats comparatifs — Requêtes image")
    lines.append("")
    lines.append("> Chaque cellule montre les 3 premières images retrouvées (score cosinus en dessous).")
    lines.append("")

    header_cols = ["Donnée client"] + [MODELS[k]["label"] for k in model_keys]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join([":---:"] * len(header_cols)) + "|")

    col_scores = {k: [] for k in model_keys}
    for user_path in demo_users:
        user_key = user_path.stem
        staged_query = _stage(str(user_path), user_key)
        query_img = _img(staged_query, user_key) + f"<br><sub>{user_key}</sub>"
        row = [query_img]
        for k in model_keys:
            entry = results.get(k, {}).get("image", {}).get(user_key, {})
            if "top5" in entry:
                col_scores[k].append(entry["top1_score"])
                row.append(_hits_cell(entry["top5"], metadata, f"{k}_{user_key}"))
            elif "error" in entry:
                row.append("erreur")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["**Score moyen Top-1**"]
    for k in model_keys:
        vals = col_scores[k]
        avg_row.append(f"**{np.mean(vals):.3f}**" if vals else "—")
    lines.append("| " + " | ".join(avg_row) + " |")
    lines.append("")

    # ── Tableau 2b : Requêtes texte ────────────────────────────────────────
    if text_model_keys:
        lines.append("## Résultats comparatifs — Requêtes texte")
        lines.append("")
        lines.append("> Modèles avec encodeur texte uniquement.")
        lines.append("")

        text_header = ["Requête"] + [MODELS[k]["label"] for k in text_model_keys]
        lines.append("| " + " | ".join(text_header) + " |")
        lines.append("|" + "|".join([":---:"] * len(text_header)) + "|")

        text_col_scores = {k: [] for k in text_model_keys}
        for query_text in TEXT_QUERIES:
            row = [f'**"{query_text}"**']
            for k in text_model_keys:
                entry = results.get(k, {}).get("text", {}).get(query_text, {})
                if "top5" in entry:
                    text_col_scores[k].append(entry["top1_score"])
                    row.append(_hits_cell(entry["top5"], metadata, f"{k}_{query_text[:20]}"))
                elif "error" in entry:
                    row.append("erreur")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        avg_text_row = ["**Score moyen Top-1**"]
        for k in text_model_keys:
            vals = text_col_scores[k]
            avg_text_row.append(f"**{np.mean(vals):.3f}**" if vals else "—")
        lines.append("| " + " | ".join(avg_text_row) + " |")
        lines.append("")

    REPORT_FILE.write_text("\n".join(lines))
    print(f"\nRapport → {REPORT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _purge_cuda()

    print("Chargement du catalogue...")
    gallery_embeddings, metadata = load_catalogue()
    print(f"  {gallery_embeddings.shape[0]:,} articles, {gallery_embeddings.shape[1]} dims")

    demo_users = sorted(DEMO_DIR.glob("user_*.png"))
    print(f"  {len(demo_users)} photos utilisateurs")

    results = {}

    for model_key in MODELS:
        print(f"\n=== {MODELS[model_key]['label']} ===")
        results[model_key] = run_model(model_key, gallery_embeddings, metadata, demo_users)

    print("\nSauvegarde des résultats...")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"  JSON → {RESULTS_FILE}")

    write_report(results, metadata, demo_users)


if __name__ == "__main__":
    main()
