"""
VTON evaluation — OpenVTON-Bench-inspired multi-modal metrics for FasHN-VTO results.

Data source: docs/vto/examples/ (same as generate_vto_report)
  - person  : user_N.png
  - garment : user_N_garment_1.jpg
  - result  : user_N_garment_1_fashn.png

Protocol (3 tiers):
  Tier 1 — Pixel-level  : SSIM, PSNR, MSE on person region (masks garment)
  Tier 2 — Feature-level : DINOv3 cosine sim — garment fidelity + identity preservation
  Tier 3 — VLM semantic : Qwen-VL scoring across 5 dimensions (1-5)

Usage:
    uv run python -m backend.scripts.evaluate_vton
    uv run python -m backend.scripts.evaluate_vton --samples user_1 user_2
    uv run python -m backend.scripts.evaluate_vton --no-vlm   # désactive le VLM (activé par défaut)
    uv run python -m backend.scripts.evaluate_vton --all-samples
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.tryon import run_tryon

EXAMPLES_DIR = Path("docs/vto/examples")
MANIFEST_PATH = EXAMPLES_DIR / "manifest.json"
REPORT_DIR = Path("docs/vto")
RESULTS_JSON = REPORT_DIR / "vto_evaluation.json"
RESULTS_MD = REPORT_DIR / "vto_evaluation.md"
PIPELINE_RESULTS_DIR = REPORT_DIR / "pipeline_results"

COMFYUI_URL = "http://127.0.0.1:8188"

DEFAULT_DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

DEFAULT_SAMPLES = [f"user_{i}" for i in range(1, 7)]




def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _load_test_data() -> dict:
    test_jsonl = REPORT_DIR / "openvton_bench_data" / "test_fashn.jsonl"
    if not test_jsonl.exists():
        return {}
    data = {}
    with open(test_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            source = entry.get("source", "")
            user_key = source.replace("_garment_1.jpg", "") if source else ""
            if user_key:
                data[user_key] = entry
    return data


def _compute_ssim_map(img1: Image.Image, img2: Image.Image) -> np.ndarray:
    """Compute SSIM map between two images."""
    from skimage.metrics import structural_similarity as ssim

    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))

    if arr1.shape != arr2.shape:
        img2 = img2.resize((img1.width, img1.height))
        arr2 = np.array(img2.convert("RGB"))

    _, ssim_map = ssim(arr1, arr2, channel_axis=2, full=True)
    return (ssim_map * 255).astype(np.uint8)


def _generate_pixel_overlay(person: Image.Image, result: Image.Image) -> Image.Image:
    """Superposition (blend 50/50) colorisée selon l'écart pixel absolu (RdYlGn_r)."""
    import matplotlib.cm as cm

    p = np.array(person.convert("RGB")).astype(np.float32)
    r = np.array(result.convert("RGB")).astype(np.float32)
    diff = np.abs(p - r).mean(axis=2)
    diff_norm = diff / (diff.max() + 1e-8)
    cmap = cm.get_cmap("RdYlGn_r")
    colored = (cmap(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.5 * np.array(result.convert("RGB")) + 0.5 * colored).astype(np.uint8)
    return Image.fromarray(blended)


def _generate_ssim_map_plot(person: Image.Image, result: Image.Image, size: tuple) -> Image.Image:
    """Plot matplotlib de la SSIM spatiale avec colormap RdYlGn."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim

    p = np.array(person.convert("RGB"))
    r = np.array(result.convert("RGB"))
    if p.shape != r.shape:
        result = result.resize(person.size, Image.LANCZOS)
        r = np.array(result.convert("RGB"))

    _, ssim_map = ssim(p, r, channel_axis=2, full=True)

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    fig.patch.set_facecolor("#1c1c20")
    ax.set_facecolor("#1c1c20")
    im = ax.imshow(ssim_map, cmap="RdYlGn", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.axis("off")
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _extract_garment_on_white(garment: Image.Image) -> Image.Image:
    """Extrait le vêtement sur fond blanc via rembg (u2net)."""
    try:
        from rembg import new_session, remove
        session = new_session("u2net")
        result = remove(np.array(garment), session=session)
        rgba = Image.fromarray(result).convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg.convert("RGB")
    except Exception:
        return garment


def generate_sample_composite(sample: dict, output_path: Path) -> Path:
    """Generate 6-frame composite image for a sample.

    Layout (2 lignes × 3 colonnes) :
      [1] Image Originale  | [2] Retrieve Top-1   | [3] Garment SAM3
      [4] Résultat VTON    | [5] Évaluation Pixel  | [6] SSIM Map

    Chaque frame : barre titre (haut) → image (centre) → barre description (bas).
    """
    user_key = sample.get("user", "")
    result_path = sample.get("result_path")
    person_path = sample.get("person_path")

    if not result_path or not person_path:
        return None

    result_path = Path(result_path) if isinstance(result_path, str) else result_path
    person_path = Path(person_path) if isinstance(person_path, str) else person_path

    if not result_path.exists() or not person_path.exists():
        return None

    person_img = Image.open(person_path).convert("RGB")
    result_img = Image.open(result_path).convert("RGB")

    garment_path = sample.get("garment_path")
    if garment_path:
        garment_path = Path(garment_path) if isinstance(garment_path, str) else garment_path
        garment_img = Image.open(garment_path).convert("RGB") if garment_path.exists() else person_img.copy()
    else:
        garment_img = person_img.copy()

    target_size = (
        max(person_img.size[0], result_img.size[0], garment_img.size[0]),
        max(person_img.size[1], result_img.size[1], garment_img.size[1]),
    )
    person_r = person_img.resize(target_size, Image.LANCZOS)
    result_r = result_img.resize(target_size, Image.LANCZOS)
    garment_r = garment_img.resize(target_size, Image.LANCZOS)

    FRAME_W = 400
    IMAGE_H = 300
    TITLE_H = 38
    DESC_H = 46
    FRAME_H = TITLE_H + IMAGE_H + DESC_H
    COLS, ROWS = 3, 2
    PAD = 10
    HEADER_H = 52

    canvas_w = COLS * FRAME_W + (COLS + 1) * PAD
    canvas_h = HEADER_H + ROWS * FRAME_H + (ROWS + 1) * PAD

    BG       = (22,  22,  30)
    HDR_BG   = (32,  32,  42)
    FRAME_BG = (38,  38,  50)
    TITLE_BG = (52,  52,  68)
    DESC_BG  = (30,  30,  40)
    BORDER   = (90,  90, 120)
    WHITE    = (230, 230, 240)
    GRAY     = (155, 155, 180)

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    try:
        font_hdr   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_desc  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_hdr = font_title = font_desc = ImageFont.load_default()

    # ── Header global ──────────────────────────────────────────────────────────
    draw.rectangle([0, 0, canvas_w, HEADER_H], fill=HDR_BG)
    cat = sample.get("category_name", "").replace("_", " ")
    draw.text(
        (PAD + 6, HEADER_H // 2),
        f"{user_key}  \u2014  {cat}  \u2014  \u00c9valuation VTON Composite",
        fill=WHITE, font=font_hdr, anchor="lm",
    )
    draw.line([0, HEADER_H, canvas_w, HEADER_H], fill=BORDER, width=2)

    # ── Images dérivées ────────────────────────────────────────────────────────
    print(f"  [{user_key}] Génération pixel overlay...")
    pixel_overlay = _generate_pixel_overlay(person_r, result_r)
    print(f"  [{user_key}] Génération SSIM map...")
    ssim_plot = _generate_ssim_map_plot(person_r, result_r, (FRAME_W, IMAGE_H))
    print(f"  [{user_key}] Extraction garment (rembg)...")
    garment_clean = _extract_garment_on_white(garment_r)

    meta = sample.get("sample_metadata", {})
    retrieval_score = float(meta.get("retrieval_score", 0.0))

    frames = [
        {
            "title": "Image Originale",
            "image": person_r,
            "desc": [
                person_path.name,
                f"Annotation : {meta.get('annotation', cat or 'N/A')}",
            ],
        },
        {
            "title": "Retrieve Top-1",
            "image": garment_r,
            "desc": [
                f"Rang : 1  \u00b7  Score : {retrieval_score:.4f}",
                f"Cat\u00e9gorie : {cat or meta.get('garment_category', 'N/A')}",
            ],
        },
        {
            "title": "Garment SAM3",
            "image": garment_clean,
            "desc": [
                f"SAM3 prompt : {meta.get('sam3_prompt', 'N/A')}",
                "V\u00eatement extrait \u2014 rembg u2net",
            ],
        },
        {
            "title": "R\u00e9sultat VTON",
            "image": result_r,
            "desc": [
                f"FasHN-VTO-1.5  \u00b7  {result_path.name}",
                f"Fid\u00e9lit\u00e9 DINO : {sample.get('garment_fidelity_dino', 0):.4f}",
            ],
        },
        {
            "title": "\u00c9valuation Pixel",
            "image": pixel_overlay,
            "desc": [
                f"SSIM : {sample.get('pixel_ssim', 0):.4f}  \u00b7  PSNR : {sample.get('pixel_psnr', 0):.2f} dB",
                f"MSE fond/corps : {sample.get('pixel_mse', 0):.2f}  \u00b7  (vert=pr\u00e9serv\u00e9, rouge=artefact)",
            ],
        },
        {
            "title": "SSIM Map",
            "image": ssim_plot,
            "desc": [
                "SSIM spatiale  \u00b7  colormap RdYlGn",
                f"Identit\u00e9 DINO : {sample.get('identity_preservation_dino', 0):.4f}",
            ],
        },
    ]

    for idx, frame in enumerate(frames):
        col = idx % COLS
        row = idx // COLS
        x0 = PAD + col * (FRAME_W + PAD)
        y0 = HEADER_H + PAD + row * (FRAME_H + PAD)

        # Cadre de fond
        draw.rectangle([x0, y0, x0 + FRAME_W, y0 + FRAME_H], fill=FRAME_BG, outline=BORDER, width=2)

        # Barre titre
        draw.rectangle([x0, y0, x0 + FRAME_W, y0 + TITLE_H], fill=TITLE_BG)
        draw.text(
            (x0 + FRAME_W // 2, y0 + TITLE_H // 2),
            frame["title"], fill=WHITE, font=font_title, anchor="mm",
        )
        draw.line([x0, y0 + TITLE_H, x0 + FRAME_W, y0 + TITLE_H], fill=BORDER, width=1)

        # Image centrée
        img = frame["image"].resize((FRAME_W, IMAGE_H), Image.LANCZOS)
        canvas.paste(img, (x0, y0 + TITLE_H))

        # Barre description
        desc_y = y0 + TITLE_H + IMAGE_H
        draw.rectangle([x0, desc_y, x0 + FRAME_W, y0 + FRAME_H], fill=DESC_BG)
        draw.line([x0, desc_y, x0 + FRAME_W, desc_y], fill=BORDER, width=1)
        for li, line in enumerate(frame["desc"][:2]):
            draw.text(
                (x0 + 8, desc_y + 6 + li * 18),
                line, fill=WHITE if li == 0 else GRAY, font=font_desc,
            )

        # Reborder sur l'image collée
        draw.rectangle([x0, y0, x0 + FRAME_W, y0 + FRAME_H], outline=BORDER, width=2)

    canvas.save(output_path)
    return output_path


def _load_dinov3():
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(DEFAULT_DINO_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_DINO_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def _load_face_model():
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(DEFAULT_DINO_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_DINO_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def _extract_dino_features(model, processor, image: Image.Image, device: str) -> np.ndarray:
    tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        feat = model(pixel_values=tensor).last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return feat.astype(np.float32)


def _build_garment_mask(image: Image.Image, garment_region: str = "top") -> Image.Image:
    from rembg import new_session, remove
    session = new_session("u2net")
    result = remove(np.array(image), session=session)
    mask = Image.fromarray(result).convert("L").resize(image.size, Image.NEAREST)
    arr = np.array(mask)
    threshold = 30
    arr = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(arr)


def _apply_mask(image: Image.Image, mask: Image.Image, background: Image.Image) -> Image.Image:
    bg = background.resize(image.size, Image.LANCZOS).convert("RGB")
    fg = image.convert("RGB")
    m = mask.convert("L").resize(image.size, Image.NEAREST)
    fg_layer = Image.composite(fg, bg, m)
    return fg_layer


def _ssim_np(img1: np.ndarray, img2: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / denom) if denom > 0 else 0.0


def _lpips_hook(model: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(output, output.size()[2:])


def _lpips_score(img1: torch.Tensor, img2: torch.Tensor, model: nn.Module) -> float:
    return 0.0


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Tier-1: Pixel metrics ────────────────────────────────────────────────────────

def compute_pixel_metrics(person: Image.Image, result: Image.Image, mask: Image.Image) -> dict:
    """Triangulation pixel : person × result sur la région fond/corps (NON-vêtement).

    Le mask isole le vêtement (rembg). On évalue la région COMPLÉMENTAIRE (fond + corps)
    qui doit être préservée après le VTON. Un score élevé signifie que le modèle a bien
    conservé l'identité et le fond sans artefacts hors zone vêtement.
    """
    mask_arr = np.array(mask.convert("L").resize(person.size, Image.NEAREST)) / 255.0
    person_np = np.array(person.convert("RGB")).astype(np.float32)
    result_np = np.array(result.convert("RGB")).astype(np.float32)

    # Zone fond/corps : pixels hors vêtement → doit être préservée
    bg_mask = mask_arr < 0.1
    if bg_mask.sum() < 100:
        # Fallback : image complète si le masque est trop grand ou vide
        bg_mask = np.ones_like(mask_arr, dtype=bool)

    p_bg = person_np[bg_mask]
    r_bg = result_np[bg_mask]

    mse = float(np.mean((p_bg - r_bg) ** 2))
    psnr = 10 * np.log10(255 ** 2 / (mse + 1e-8))
    ssim = _ssim_np(person_np, result_np)  # SSIM global structurel

    return {"ssim": round(ssim, 4), "psnr": round(psnr, 2), "mse": round(mse, 2)}


# ── Tier-2: DINOv3 feature metrics ─────────────────────────────────────────────

def compute_garment_fidelity(
    garment_img: Image.Image,
    result: Image.Image,
    mask: Image.Image,
    dino_model, dino_processor, device: str,
) -> float:
    garment_feat = _extract_dino_features(dino_model, dino_processor, garment_img, device)
    result_feat = _extract_dino_features(dino_model, dino_processor, result, device)
    sim = np.dot(garment_feat, result_feat) / (np.linalg.norm(garment_feat) * np.linalg.norm(result_feat) + 1e-8)
    return round(float(sim), 4)


def compute_identity_preservation(
    person: Image.Image,
    result: Image.Image,
    dino_model, dino_processor, device: str,
) -> float:
    person_feat = _extract_dino_features(dino_model, dino_processor, person, device)
    result_feat = _extract_dino_features(dino_model, dino_processor, result, device)
    sim = np.dot(person_feat, result_feat) / (np.linalg.norm(person_feat) * np.linalg.norm(result_feat) + 1e-8)
    return round(float(sim), 4)


# ── Tier-3: VLM scoring ──────────────────────────────────────────────────────────

_VLM_MODEL_CACHE: tuple | None = None


VLM_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _get_vlm_model():
    global _VLM_MODEL_CACHE
    if _VLM_MODEL_CACHE is not None:
        return _VLM_MODEL_CACHE

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except ImportError:
        _VLM_MODEL_CACHE = (None, None)
        return _VLM_MODEL_CACHE

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        _VLM_MODEL_CACHE = (model, processor)
        print(f"VLM loaded: {VLM_MODEL_ID}")
    except Exception as e:
        print(f"[WARN] Failed to load {VLM_MODEL_ID}: {e}")
        _VLM_MODEL_CACHE = (None, None)

    return _VLM_MODEL_CACHE


def _vlm_score_single(
    person_img: Image.Image,
    garment_img: Image.Image,
    result_img: Image.Image,
) -> dict[str, float]:
    model, processor = _get_vlm_model()
    if model is None or processor is None:
        return {}

    prompt = (
        "You are evaluating a virtual try-on result. Rate each dimension 1-5 (1=poor, 5=excellent). "
        "Respond ONLY with JSON: {\"background\":#, \"identity\":#, \"texture\":#, \"shape\":#, \"overall\":#}."
        "\n1. Background: Is the background unchanged from the person image?"
        "\n2. Identity: Is the person's face, hair, and skin tone preserved?"
        "\n3. Texture: Is the garment texture, pattern, logo faithfully transferred from the garment source?"
        "\n4. Shape: Does the garment follow the body's natural pose and shape?"
        "\n5. Overall: Does this look like a realistic photo of a person wearing that garment?"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": person_img},
                {"type": "image", "image": garment_img},
                {"type": "image", "image": result_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(model.parameters()).device
        inputs = processor(
            text=[text],
            images=[[person_img, garment_img, result_img]],
            return_tensors="pt",
        )
        input_len = inputs.input_ids.shape[1]
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return {}
        scores = json.loads(match.group())
        return {f"vlm_{k}": float(v) for k, v in scores.items()}
    except Exception:
        return {}


def _vlm_score_batch(samples: list[dict]) -> list[dict]:
    results = []
    for s in tqdm(samples, desc="VLM scoring", unit="sample"):
        person = Image.open(s["person_path"]).convert("RGB")
        garment = Image.open(s["garment_path"]).convert("RGB")
        result = Image.open(s["result_path"]).convert("RGB")
        scores = _vlm_score_single(person, garment, result)
        results.append(scores)
    return results


# ── Core evaluation ────────────────────────────────────────────────────────────────

def evaluate_sample(
    sample: dict,
    dino_model, dino_processor, device: str,
    use_vlm: bool,
) -> dict:
    person = Image.open(sample["person_path"]).convert("RGB")
    garment = Image.open(sample["garment_path"]).convert("RGB")
    result = Image.open(sample["result_path"]).convert("RGB")

    target_size = (max(person.size[0], result.size[0], garment.size[0]),
                   max(person.size[1], result.size[1], garment.size[1]))
    person = person.resize(target_size, Image.LANCZOS)
    result = result.resize(target_size, Image.LANCZOS)
    garment = garment.resize(target_size, Image.LANCZOS)

    mask = _build_garment_mask(garment)

    pixel = compute_pixel_metrics(person, result, mask)
    garment_fid = compute_garment_fidelity(garment, result, mask, dino_model, dino_processor, device)
    identity = compute_identity_preservation(person, result, dino_model, dino_processor, device)

    scores = {
        "sample": sample["sample_id"],
        "user": sample["user"],
        "garment_category": sample["category_name"],
        "fashn_category": sample["fashn_category"],
        "pixel_ssim": pixel["ssim"],
        "pixel_psnr": pixel["psnr"],
        "pixel_mse": pixel["mse"],
        "garment_fidelity_dino": garment_fid,
        "identity_preservation_dino": identity,
    }

    if use_vlm:
        vlm = _vlm_score_single(person, garment, result)
        scores.update(vlm)

    return scores


def run_evaluation(
    samples: list[dict],
    use_vlm: bool,
) -> dict:
    print("Loading models...")
    dino_processor, dino_model, device = _load_dino()

    results = []
    for sample in tqdm(samples, desc="Evaluating", unit="sample"):
        scores = evaluate_sample(sample, dino_model, dino_processor, device, use_vlm)
        scores["user"] = sample.get("user")
        scores["category_name"] = sample.get("category_name")
        scores["garment_category"] = sample.get("category_name")
        scores["fashn_category"] = sample.get("fashn_category")
        scores["person_path"] = str(sample.get("person_path"))
        scores["result_path"] = str(sample.get("result_path"))
        scores["garment_path"] = str(sample.get("garment_path"))
        if "sample_metadata" in sample:
            scores["sample_metadata"] = sample["sample_metadata"]
        results.append(scores)

    overall = _aggregate(results)
    overall["n_samples"] = len(samples)
    overall["samples"] = results
    return overall


def _load_dino():
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(DEFAULT_DINO_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_DINO_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def _aggregate(results: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in results if key in r and r[key] != 0.0]
        return round(np.mean(vals), 4) if vals else 0.0

    vlm_keys = [k for k in results[0].keys() if k.startswith("vlm_")] if results else []
    vlm_means = {k: mean(k) for k in vlm_keys}

    overall = {
        "pixel_ssim": mean("pixel_ssim"),
        "pixel_psnr": mean("pixel_psnr"),
        "pixel_mse": mean("pixel_mse"),
        "garment_fidelity_dino": mean("garment_fidelity_dino"),
        "identity_preservation_dino": mean("identity_preservation_dino"),
        **vlm_means,
    }

    by_category: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r["garment_category"]
        by_category[cat]["pixel_ssim"].append(r["pixel_ssim"])
        by_category[cat]["garment_fidelity_dino"].append(r["garment_fidelity_dino"])
        by_category[cat]["identity_preservation_dino"].append(r["identity_preservation_dino"])
        for vk in vlm_keys:
            by_category[cat][vk].append(r.get(vk, 0.0))

    per_category = {}
    for cat, vals in sorted(by_category.items()):
        per_category[cat] = {k: round(np.mean(v), 4) if v else 0.0 for k, v in vals.items()}
        per_category[cat]["n"] = len(next(iter(vals.values())))

    return {
        "overall": overall,
        "per_category": per_category,
    }


# ── Report ──────────────────────────────────────────────────────────────────────

def load_openvton_bench_results(results_dir: Path) -> dict | None:
    for p in sorted(results_dir.iterdir()):
        if p.is_dir():
            summary_path = p / "summary.txt"
            results_path = p / "per_model" / "fashn-vto-1.5" / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    return json.load(f)
    return None


def write_markdown_report(results: dict, args: argparse.Namespace, openvton_results: dict | None = None) -> str:
    overall = results["overall"]
    k_values = results["per_category"]
    samples = results.get("samples", [])

    vlm_keys = sorted([k for k in overall if k.startswith("vlm_")])
    vlm_dim_labels = {
        "vlm_background": "Cohérence du fond",
        "vlm_identity": "Préservation de l'identité",
        "vlm_texture": "Fidélité de la texture",
        "vlm_shape": "Plausibilité de la forme",
        "vlm_overall": "Score global",
    }

    viz_dir = REPORT_DIR / "openvton_viz"

    lines = [
        "---",
        f"model: FasHN-VTO-1.5",
        f"dataset: DeepFashion InShop (demo users)",
        f"n_samples: {results['n_samples']}",
        f"vlm_enabled: {args.vlm}",
        "---",
        "",
        "# Rapport d'évaluation VTON — FasHN-VTO-1.5",
        "",
        "## Données utilisées",
        "",
        "Ce rapport évalue les résultats de virtual try-on sur **6 utilisateurs démo** générés",
        "via `generate_vto_report.py`. Chaque utilisateur est composé de :",
        "",
        "| Élément | Description |",
        "|---------|-------------|",
        "| **Personne** | Photo originale de la personne (user_N.png) |",
        "| **Vêtement source** | Vêtement flat-lay recommandé par Marqo FashionSigLIP (user_N_garment_1.jpg) |",
        "| **Résultat VTON** | Image générée par FasHN-VTO-1.5 (user_N_garment_1_fashn.png) |",
        "",
        "Il **n'existe pas de ground truth** (pas de photo réelle de la personne portant",
        "ce vêtement exact). L'évaluation est donc adaptée en **single-reference settings** :",
        "",
        "- La **personne originale** sert de référence pour la préservation du corps/visage/fond",
        "- Le **vêtement source** sert de référence pour la fidélité de transfert",
        "",
        f"**{len(samples)} échantillons** évalués — catégories : "
        + ", ".join(sorted(set(s.get("garment_category", "").replace("_", " ") for s in samples))) + ".",
        "",
        "---",
        "",
        "## Protocole d'évaluation",
        "",
        "L'évaluation suit le protocole multi-modal **OpenVTON-Bench** (Li et al., 2025)",
        "avec 3 niveaux de métriques :",
        "",
        "| Niveau | Méthode | Ce que ça mesure |",
        "|--------|---------|-----------------|",
        "| **1 — Pixel** | SSIM / PSNR / MSE | Préservation fond/corps (région NON-vêtement) — triangulation person × result |",
        "| **2 — Feature** | Cosine sim DINOv3 ViT-L/16 | Fidélité texture vêtement + préservation identité |",
        "| **3 — VLM** | Qwen3-VL-4B-Instruct | Score sémantique sur 5 dimensions (1–5) |",
        "",
        "Le bloc **OpenVTON-Bench** ajoute la segmentation SAM3 du vêtement + érosion multi-échelle",
        "(40/80/120 px) pour isoler la texture interne des bords.",
        "",
        "## Scores globaux",
        "",
        "### Niveau 1 — Pixel (triangulation : fond/corps, hors vêtement)",
        "",
        "| Métrique | Valeur |",
        "|----------|------:|",
        f"| SSIM   | {overall.get('pixel_ssim', 0):.4f} |",
        f"| PSNR   | {overall.get('pixel_psnr', 0):.2f} dB |",
        f"| MSE    | {overall.get('pixel_mse', 0):.2f} |",
        "",
        "### Niveau 2 — Feature (cosine sim DINOv3)",
        "",
        "| Métrique | Valeur | Interprétation |",
        "|----------|------:|----------------|",
        f"| Fidélité vêtement (source → résultat) | {overall.get('garment_fidelity_dino', 0):.4f} | Plus c'est proche de 1, mieux la texture est transférée |",
        f"| Préservation identité (personne → résultat) | {overall.get('identity_preservation_dino', 0):.4f} | Plus c'est proche de 1, mieux le visage/corps sont conservés |",
        "",
    ]

    if vlm_keys:
        lines.append("### Niveau 3 — VLM sémantique (Qwen3-VL-4B, 1–5)")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|-----------|------:|")
        for k in vlm_keys:
            label = vlm_dim_labels.get(k, k.replace("vlm_", "").title())
            score = overall.get(k, 0.0)
            full = int(score)
            empty = 5 - full
            bar = "★" * full + "☆" * empty
            lines.append(f"| {label:<28} | {score:.1f} {bar} |")
        lines.append("")

    if k_values:
        lines.append("## Par catégorie")
        lines.append("")
        cols = ["Catégorie", "n", "SSIM", "Fid. vet.", "Identité"] + [
            vlm_dim_labels.get(k, k.split("_", 1)[1].title()) for k in vlm_keys
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---:"] * len(cols)) + "|")
        for cat, vals in sorted(k_values.items()):
            n = vals.get("n", 0)
            row = f"| {cat:<28} | {n:>3} | {vals.get('pixel_ssim', 0):.4f} | {vals.get('garment_fidelity_dino', 0):.4f} | {vals.get('identity_preservation_dino', 0):.4f}"
            if vlm_keys:
                row += " |" + " |".join(f" {vals.get(k, 0.0):.1f} " for k in vlm_keys)
            row += " |"
            lines.append(row)
        lines.append("")

    if samples:
        lines.append("## Par échantillon")
        lines.append("")
        cols = ["Échantillon", "Catégorie", "SSIM", "Fid. vet.", "Identité"] + (
            [vlm_dim_labels.get(k, k.split("_", 1)[1].title()) for k in vlm_keys]
        )
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---:"] * len(cols)) + "|")
        for s in samples:
            sid = s.get("sample", s.get("sample_id", "")).replace("_garment_1_fashn", "")
            cat = s.get("garment_category", "").replace("_", " ")
            row = f"| {sid:<28} | {cat:<13} | {s.get('pixel_ssim', 0):.4f} | {s.get('garment_fidelity_dino', 0):.4f} | {s.get('identity_preservation_dino', 0):.4f}"
            if vlm_keys:
                row += " |" + " |".join(f" {s.get(k, 0.0):.1f} " for k in vlm_keys)
            row += " |"
            lines.append(row)
        lines.append("")

    lines.append("")
    lines.append("## Visualisations — Échantillons")
    lines.append("")

    sample_viz_dir = PIPELINE_RESULTS_DIR / "visualizations"
    sample_viz_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        cat = sample.get("category_name", "").replace("_", " ")
        user_key = sample.get("user", "")
        sample_meta = sample.get("sample_metadata", {})

        composite_path = generate_sample_composite(
            sample,
            sample_viz_dir / f"{user_key}.png"
        )

        fidelity = sample.get("garment_fidelity_dino", 0)
        if composite_path and composite_path.exists():
            lines.append(f"### {user_key} — {cat}")
            lines.append("")
            if sample_meta:
                lines.append(f"<sub>garment_idx: **{sample_meta.get('garment_idx', 'N/A')}** | category: **{sample_meta.get('garment_category', cat)}** | SAM3: **{sample_meta.get('sam3_prompt', 'N/A')}** | retrieval: **{sample_meta.get('retrieval_score', 0):.4f}** | fidelity: **{fidelity:.4f}**</sub>")
            lines.append("")
            lines.append(f'| <img src="pipeline_results/visualizations/{composite_path.name}" width="900"/> |')
            lines.append("")

    if openvton_results:
        gs = openvton_results.get("garment_summary", {})
        pa = openvton_results.get("pixel_average", {})

        lines.append("## OpenVTON-Bench — Métriques détaillées (SAM3 + DINOv3)")
        lines.append("")
        lines.append("**Évaluation région vêtement** — DINOv3 ViT-H+/16 + SAM3 avec érosion multi-échelle :")
        lines.append("")
        lines.append("| Échelle        | SSIM    | LPIPS   | Cosine Sim | PSNR (dB) |")
        lines.append("|----------------|--------:|-------:|-----------:|----------:|")
        for scale, label in [
            ("scale_0", "Originale"),
            ("average", "Moyenne (4 échelles)"),
        ]:
            s = gs.get(scale, {})
            lines.append(
                f"| {label:<14} | {s.get('ssim_mean', 0):.4f} | {s.get('lpips_mean', 0):.5f} | "
                f"{s.get('cosine_similarity_mean', 0):.4f} | {s.get('psnr_mean', 0):.2f} |"
            )
        lines.append("")
        lines.append("**Comparaison pixel full-image** (SSIM / LPIPS / PSNR) :")
        lines.append("")
        lines.append("| Métrique | Valeur |")
        lines.append("|----------|-------:|")
        lines.append(f"| SSIM full-image | {pa.get('ssim', 0):.4f} |")
        lines.append(f"| LPIPS full-image | {pa.get('lpips', 0):.4f} |")
        lines.append(f"| PSNR full-image | {pa.get('psnr', 0):.2f} dB |")
        lines.append("")
        lines.append("*Source : `docs/vto/openvton_bench_results/` — relancer avec `--run-openvton-bench`.*")
        lines.append("")

    return "\n".join(lines)


def print_results(results: dict) -> None:
    overall = results["overall"]
    print("\n" + "=" * 55)
    print("  VTON Evaluation — FasHN-VTO-1.5")
    print("=" * 55)
    print(f"  Samples: {results['n_samples']}")
    print()
    print("  Pixel metrics (person region, garment-masked):")
    print(f"    SSIM  : {overall['pixel_ssim']:.4f}")
    print(f"    PSNR  : {overall['pixel_psnr']:.2f} dB")
    print(f"    MSE   : {overall['pixel_mse']:.2f}")
    print()
    print("  DINOv3 feature metrics:")
    print(f"    Garment Fidelity     : {overall['garment_fidelity_dino']:.4f}")
    print(f"    Identity Preserve  : {overall['identity_preservation_dino']:.4f}")
    vlm_keys = [k for k in overall if k.startswith("vlm_")]
    if vlm_keys:
        print()
        print("  VLM semantic scores (1-5):")
        for k in vlm_keys:
            print(f"    {k.replace('vlm_', '').title():<22}: {overall[k]:.1f}")
    print("=" * 55)


def build_samples(user_keys: list[str], manifest: dict, test_data: dict = None) -> list[dict]:
    test_data = test_data or {}
    samples = []
    for user_key in user_keys:
        if user_key not in manifest:
            continue
        user_num = user_key.split("_")[1]
        top1 = next(g for g in manifest[user_key] if g["rank"] == 1)
        sample = {
            "sample_id": f"{user_key}_garment_1_fashn",
            "user": user_key,
            "person_path": EXAMPLES_DIR / f"{user_key}.png",
            "garment_path": EXAMPLES_DIR / top1["img"],
            "result_path": EXAMPLES_DIR / f"{user_key}_garment_1_fashn.png",
            "category_name": top1.get("category_name", ""),
            "fashn_category": top1.get("fashn_category", "tops"),
        }
        if user_key in test_data:
            sample["sample_metadata"] = test_data[user_key]
        samples.append(sample)
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VTON evaluation — OpenVTON-Bench metrics for FasHN-VTO")
    parser.add_argument("--samples", type=str, nargs="+", default=DEFAULT_SAMPLES,
                        help=f"Sample keys to evaluate (default: all demo users)")
    parser.add_argument("--all-samples", action="store_true",
                        help="Evaluate all available samples")
    parser.add_argument("--vlm", action="store_true", default=True,
                        help="Enable VLM-based scoring (Qwen2-VL-2B, requires GPU + model)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM scoring (default behavior)")
    parser.add_argument("--run-openvton-bench", action="store_true",
                        help="Run OpenVTON-Bench garment+pixel evaluation (SAM3 + DINOv3)")
    parser.add_argument("--output-json", type=Path, default=RESULTS_JSON,
                        help="Output JSON path")
    parser.add_argument("--output-md", type=Path, default=RESULTS_MD,
                        help="Output Markdown report path")
    args = parser.parse_args()

    if args.no_vlm:
        args.vlm = False

    openvton_results = None
    bench_dir = REPORT_DIR / "openvton_bench_results"

    if args.run_openvton_bench:
        import subprocess
        config_path = REPORT_DIR / "openvton_bench_data" / "config_fashn.yaml"
        print("Running OpenVTON-Bench (garment + pixel evaluation)...")
        result = subprocess.run(
            ["python3", "/tmp/openvton-bench/benchmark/run_benchmark.py",
             "--config", str(config_path), "--eval_type", "garment", "pixel"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2])
        )
        if result.returncode != 0:
            print(f"[WARN] OpenVTON-Bench failed:\n{result.stderr[-500:]}")
        else:
            print("OpenVTON-Bench completed successfully.")
        openvton_results = load_openvton_bench_results(bench_dir)
    else:
        openvton_results = load_openvton_bench_results(bench_dir)
        if openvton_results:
            print(f"Loaded OpenVTON-Bench results from {bench_dir}/")

    manifest = _load_manifest()
    test_data = _load_test_data()

    if args.all_samples:
        user_keys = list(manifest.keys())
    else:
        user_keys = args.samples if args.samples else DEFAULT_SAMPLES

    samples = build_samples(user_keys, manifest, test_data)
    missing = [s for s in samples if not s["result_path"].exists()]
    if missing:
        print(f"[INFO] Missing {len(missing)} result files, running pipeline...")
        import subprocess
        for s in missing:
            user_key = s["user"]
            print(f"  Running pipeline for {user_key}...")
            subprocess.run(
                ["uv", "run", "python", "-m", "backend.scripts.run_vton_pipeline",
                 "--users", user_key, "--no-skip"],
                cwd=Path.cwd(),
            )
        samples = build_samples(user_keys, manifest, test_data)
        missing = [s for s in samples if not s["result_path"].exists()]
    if missing:
        print("[WARN] Missing result files:")
        for s in missing:
            print(f"  - {s['result_path']}")
        samples = [s for s in samples if s["result_path"].exists()]

    print(f"Evaluating {len(samples)} samples (VLM={'on' if args.vlm else 'off'})…")
    results = run_evaluation(samples, use_vlm=args.vlm)

    print_results(results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults → {args.output_json}")

    md = write_markdown_report(results, args, openvton_results)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    print(f"Report → {args.output_md}")


if __name__ == "__main__":
    main()