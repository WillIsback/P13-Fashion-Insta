"""
Generate VTO result images and docs/vto/vto_comparison.md report.

Pipeline for each demo user:
  1. Retrieve top-3 garments via Marqo FashionSigLIP
  2. Copy garment images to docs/vto/examples/
  3. Run VTO (flux, fashn, qwen) against top-1 garment
  4. Write markdown report

Usage:
    uv run python -m backend.scripts.generate_vto_report

Prerequisites:
    - ComfyUI running at http://127.0.0.1:8188
    - Marqo index built: uv run python -m backend.scripts.build_clip_indices --model marqo_fashion_siglip
    - All 3 workflow JSON templates present in comfyui_api/
"""
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.retrieval import recommend
from backend.core.tryon import build_qwen_prompt, run_tryon

EMBEDDINGS = Path("data/embeddings_marqo_fashion_siglip.npy")
METADATA = Path("data/index_metadata.json")
DEMO_DIR = Path("demo")
OUT_DIR = Path("docs/vto/examples")
REPORT_FILE = Path("docs/vto/vto_comparison.md")
MANIFEST = OUT_DIR / "manifest.json"

COMFYUI_URL = "http://127.0.0.1:8188"
TOP_GARMENTS = 3

WORKFLOWS = {
    "flux":  Path("comfyui_api/image_flux2_klein_image_edit_4b_base_api.json"),
    "fashn": Path("comfyui_api/FasHN-VTO_api.json"),
    "qwen":  Path("comfyui_api/image_qwen_image_edit_2511_api.json"),
}

POLL_TIMEOUTS = {
    "flux":  120.0,
    "fashn": 180.0,
    "qwen":  600.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_comfyui():
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=3).raise_for_status()
    except Exception:
        print(f"[ERROR] Cannot reach ComfyUI at {COMFYUI_URL}. Start it first.")
        sys.exit(1)


def _interrupt_and_clear():
    try:
        requests.post(f"{COMFYUI_URL}/interrupt", timeout=5)
        requests.post(f"{COMFYUI_URL}/queue", json={"clear": True}, timeout=5)
        time.sleep(2)
    except Exception:
        pass


# ── Step 1 : Retrieval ─────────────────────────────────────────────────────────

def run_retrieval(demo_users: list[Path]) -> dict:
    """
    Retrieve top-3 garments per user with Marqo FashionSigLIP.
    Copies garment images to OUT_DIR and returns the manifest dict.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for user_path in demo_users:
        user_key = user_path.stem   # "user_1"
        user_num = user_key.split("_")[1]

        # Copy user photo
        dest_user = OUT_DIR / user_path.name
        if not dest_user.exists():
            shutil.copy2(user_path, dest_user)

        user_img = Image.open(user_path).convert("RGB")
        results = recommend(user_img, EMBEDDINGS, METADATA, top_n=TOP_GARMENTS)

        garments = []
        for rank, r in enumerate(results, start=1):
            meta = r["metadata"]
            src = Path(meta["path"])
            dest = OUT_DIR / f"user_{user_num}_garment_{rank}{src.suffix}"
            if not dest.exists():
                shutil.copy2(src, dest)

            garments.append({
                "rank": rank,
                "img": dest.name,
                "score": round(r["score"], 4),
                "category_name": meta.get("category_name", ""),
                "fashn_category": meta.get("fashn_category", "tops"),
                "archetype": meta.get("archetype", ""),
            })
            print(f"  {user_key} #{rank}: {meta.get('category_name')} ({r['score']:.3f})")

        manifest[user_key] = garments

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {MANIFEST}")
    return manifest


# ── Step 2 : VTO ──────────────────────────────────────────────────────────────

def run_vto(manifest: dict) -> None:
    """Run VTO for each user's top-1 garment across all 3 workflows."""
    for user_key, garments in manifest.items():
        user_num = user_key.split("_")[1]
        user_img = Image.open(OUT_DIR / f"user_{user_num}.png").convert("RGB")
        top1 = next(g for g in garments if g["rank"] == 1)
        garment_img = Image.open(OUT_DIR / top1["img"]).convert("RGB")

        for wf_name, wf_path in WORKFLOWS.items():
            out_path = OUT_DIR / f"user_{user_num}_garment_1_{wf_name}.png"
            if out_path.exists():
                print(f"  [skip] {out_path.name}")
                continue

            print(f"  {wf_name} · {user_key} / {top1['category_name']}…", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_tryon(
                    user_img=user_img,
                    item_img=garment_img,
                    colour="",
                    category=top1["fashn_category"],
                    prompt=build_qwen_prompt(top1),
                    workflow=wf_name,
                    comfyui_url=COMFYUI_URL,
                    api_template_path=wf_path,
                    poll_timeout=POLL_TIMEOUTS[wf_name],
                )
                result.save(out_path)
                print(f"done ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                _interrupt_and_clear()


# ── Step 3 : Report ────────────────────────────────────────────────────────────

def write_report(manifest: dict) -> None:
    lines = []

    lines.append("# Virtual Try-On — Provider Comparison Report")
    lines.append("")
    lines.append(f"**Date:** {date.today()}  ")
    lines.append(f"**Catalogue:** DeepFashion InShop  ")
    lines.append(f"**Queries:** {len(manifest)} demo user photos  ")
    lines.append("**Retrieval model:** Marqo FashionSigLIP (768-dim, image + text)  ")
    lines.append("**VTO providers:** Flux2-Klein · FasHN-VTO · Qwen-Image-Edit-2511  ")
    lines.append("")
    lines.append("> VTO images generated by running `uv run python -m backend.scripts.generate_vto_report` with ComfyUI running on `http://127.0.0.1:8188`.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Providers")
    lines.append("")
    lines.append("| Key | Model family | Approach | Input |")
    lines.append("|-----|-------------|----------|-------|")
    lines.append("| **Flux2-Klein** | MMDiT (FLUX.1) | Image editing via inpainting | Person + garment + colour prompt |")
    lines.append("| **FasHN-VTO** | MMDiT (specialised) | Dedicated try-on inference | Person + garment + category |")
    lines.append("| **Qwen-Image-Edit-2511** | MMDiT (Qwen-VL 7B) | Instruction-following image edit | Person + garment + text prompt |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for user_key, garments in manifest.items():
        user_num = user_key.split("_")[1]
        top1 = next(g for g in garments if g["rank"] == 1)

        lines.append(f"## User {user_num}")
        lines.append("")

        # Retrieval table
        score_headers = " | ".join(f"Garment #{g['rank']} (score: {g['score']})" for g in garments)
        lines.append("<table>")
        lines.append("<tr>")
        lines.append("<th>User photo</th>")
        for g in garments:
            lines.append(f"<th>Garment #{g['rank']} (score: {g['score']})</th>")
        lines.append("</tr>")
        lines.append("<tr>")
        lines.append(f'<td><img src="examples/user_{user_num}.png" width="160"/></td>')
        for g in garments:
            lines.append(
                f'<td><img src="examples/{g["img"]}" width="160"/>'
                f'<br/><sub>{g["category_name"]} · {g["fashn_category"]}'
                + (f'<br/>{g["archetype"]}' if g.get("archetype") else "")
                + '</sub></td>'
            )
        lines.append("</tr>")
        lines.append("</table>")
        lines.append("")

        # VTO table
        lines.append(f"### Try-On — Top-1 Garment ({top1['category_name']})")
        lines.append("")
        lines.append("| Flux2-Klein | FasHN-VTO | Qwen-Image-Edit-2511 |")
        lines.append("|:-----------:|:---------:|:--------------------:|")
        vto_imgs = " | ".join(
            f'<img src="examples/user_{user_num}_garment_1_{wf}.png" width="200"/>'
            for wf in ["flux", "fashn", "qwen"]
        )
        lines.append(f"| {vto_imgs} |")
        lines.append(f"| Colour prompt | Category: {top1['fashn_category']} | Auto-prompt from annotation |")
        lines.append("")
        lines.append("---")
        lines.append("")

    REPORT_FILE.write_text("\n".join(lines))
    print(f"Report → {REPORT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not EMBEDDINGS.exists():
        print(f"[ERROR] Marqo index not found: {EMBEDDINGS}")
        print("  Run: uv run python -m backend.scripts.build_clip_indices --model marqo_fashion_siglip")
        sys.exit(1)

    _check_comfyui()

    demo_users = sorted(DEMO_DIR.glob("user_*.png"))
    print(f"Found {len(demo_users)} demo users\n")

    print("=== Step 1: Retrieval (Marqo FashionSigLIP) ===")
    manifest = run_retrieval(demo_users)

    print("\n=== Step 2: VTO ===")
    run_vto(manifest)

    print("\n=== Step 3: Report ===")
    write_report(manifest)


if __name__ == "__main__":
    main()
