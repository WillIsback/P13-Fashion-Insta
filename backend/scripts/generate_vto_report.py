"""
Generate VTO result images for the docs/vto/vto_comparison.md report.

Runs retrieval + 3-provider VTO (flux, fashn, qwen) for each demo user,
saving results to docs/vto/examples/ as:
    user_{N}_garment_1_{provider}.png

Usage:
    uv run python -m backend.scripts.generate_vto_report

Prerequisites:
    - ComfyUI running at http://127.0.0.1:8188
    - Catalogue index built: uv run python -m backend.scripts.build_index
    - All 3 workflow JSON templates present in phase_2/
"""
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.tryon import build_qwen_prompt, run_tryon

EMBEDDINGS = Path("data/embeddings.npy")
METADATA = Path("data/index_metadata.json")
DEMO_DIR = Path("demo")
OUT_DIR = Path("docs/vto/examples")
MANIFEST = OUT_DIR / "manifest.json"

COMFYUI_URL = "http://127.0.0.1:8188"

WORKFLOWS = {
    "flux":  Path("phase_2/tryon_api.json"),
    "fashn": Path("phase_2/FasHN-VTO_api.json"),
    "qwen":  Path("phase_2/image_qwen_image_edit_2511_api.json"),
}

# Qwen loads a 7.9 GB text encoder on first use — needs much more time.
POLL_TIMEOUTS = {
    "flux":  120.0,
    "fashn": 180.0,
    "qwen":  600.0,
}


def _check_comfyui():
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=3).raise_for_status()
    except Exception:
        print(f"[ERROR] Cannot reach ComfyUI at {COMFYUI_URL}. Start it first.")
        sys.exit(1)


def _interrupt_and_clear():
    """Send interrupt + clear queue so a timed-out job doesn't block the GPU."""
    try:
        requests.post(f"{COMFYUI_URL}/interrupt", timeout=5)
        requests.post(f"{COMFYUI_URL}/queue", json={"clear": True}, timeout=5)
        time.sleep(2)  # give ComfyUI time to unload
    except Exception:
        pass


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        print("[ERROR] manifest.json not found. Run retrieval first.")
        sys.exit(1)
    return json.loads(MANIFEST.read_text())


def _run_one(user_img: Image.Image, garment_img: Image.Image, garment_meta: dict, workflow: str) -> Image.Image:
    return run_tryon(
        user_img=user_img,
        item_img=garment_img,
        colour="",
        category=garment_meta.get("fashn_category", "tops"),
        prompt=build_qwen_prompt(garment_meta),
        workflow=workflow,
        comfyui_url=COMFYUI_URL,
        api_template_path=WORKFLOWS[workflow],
        poll_timeout=POLL_TIMEOUTS[workflow],
    )


def main():
    _check_comfyui()
    manifest = _load_manifest()

    for user_key, garments in manifest.items():
        user_num = user_key.split("_")[1]
        user_path = DEMO_DIR / f"user_{user_num}.png"
        user_img = Image.open(user_path).convert("RGB")

        # Only VTO against top-1 garment (rank=1)
        top1 = next(g for g in garments if g["rank"] == 1)
        garment_img = Image.open(OUT_DIR / top1["img"]).convert("RGB")

        for wf_name in WORKFLOWS:
            out_path = OUT_DIR / f"user_{user_num}_garment_1_{wf_name}.png"
            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                continue

            print(f"  Running {wf_name} for {user_key} / {top1['category_name']}…", end=" ", flush=True)
            t0 = time.time()
            try:
                result = _run_one(user_img, garment_img, top1, wf_name)
                result.save(out_path)
                print(f"done ({time.time() - t0:.1f}s) → {out_path.name}")
            except Exception as e:
                print(f"FAILED: {e}")
                _interrupt_and_clear()

    print("\nAll VTO images generated. Open docs/vto/vto_comparison.md to view the report.")


if __name__ == "__main__":
    main()
