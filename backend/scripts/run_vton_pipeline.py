"""
E2E VTON Pipeline:
  1. Retrieve: Pour chaque user dans demo/, utiliser retrieval pour trouver top garment
  2. Try-on: Appeler FasHN-VTO via ComfyUI
  3. Save: Sauvegarder résultat + annotations
  4. Evaluate: Métriques VTON

Usage:
    uv run python -m backend.scripts.run_vton_pipeline --users user_1 user_2
    uv run python -m backend.scripts.run_vton_pipeline --all
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from backend.core.embedder import embed_image
from backend.core.retrieval import embed_query, recommend, search
from backend.core.tryon import run_tryon
from backend.scripts.evaluate_vton import (
    EXAMPLES_DIR,
    _load_dino,
    _load_dinov3,
    _load_face_model,
    evaluate_sample,
    run_evaluation,
)

# Paths
DEMO_DIR = Path("demo")
DATA_DIR = Path("data")
CATALOGUE_EMBEDDINGS = DATA_DIR / "embeddings_marqo_fashion_siglip.npy"
CATALOGUE_METADATA = DATA_DIR / "index_metadata.json"
REPORT_DIR = Path("docs/vto")
PIPELINE_RESULTS_DIR = REPORT_DIR / "pipeline_results"
EXAMPLES_DIR = REPORT_DIR / "examples"

MANIFEST_PATH = EXAMPLES_DIR / "manifest.json"


def load_manifest() -> dict:
    """Load manifest with pre-computed retrieval results."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def load_catalogue_metadata() -> list[dict]:
    """Load catalogue metadata."""
    if not CATALOGUE_METADATA.exists():
        raise FileNotFoundError(f"Catalogue metadata not found: {CATALOGUE_METADATA}")
    with open(CATALOGUE_METADATA) as f:
        return json.load(f)


def load_user_image(user_key: str) -> Image.Image:
    """Load user photo from demo directory."""
    user_path = DEMO_DIR / f"{user_key}.png"
    if not user_path.exists():
        raise FileNotFoundError(f"User image not found: {user_path}")
    return Image.open(user_path).convert("RGB")


def retrieve_top_garment(
    user_img: Image.Image,
    embeddings_path: Path,
    metadata_path: Path,
    top_n: int = 1,
) -> list[dict]:
    """Retrieve top N garments for a user image."""
    query_vec = embed_query(user_img)
    return search(query_vec, embeddings_path, metadata_path, top_n=top_n)


def run_pipeline(
    user_keys: list[str],
    top_n: int = 1,
    skip_existing: bool = True,
    run_evaluation_flag: bool = True,
) -> list[dict]:
    """Exécute le pipeline complet pour tous les utilisateurs."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    catalogue_meta = load_catalogue_metadata()

    print(f"Pipeline E2E pour {len(user_keys)} utilisateur(s)")

    if run_evaluation_flag:
        print("Loading evaluation models...")
        dino_processor, dino_model, device = _load_dino()
    else:
        dino_processor = dino_model = device = None

    all_results = []

    for user_key in tqdm(user_keys, desc="Pipeline"):
        print(f"\n{'='*50}")
        print(f"Traitement: {user_key}")
        print(f"{'='*50}")

        result_file = PIPELINE_RESULTS_DIR / f"{user_key}_result.json"
        if skip_existing and result_file.exists():
            print(f"Skipping {user_key} (already exists)")
            with open(result_file) as f:
                user_result = json.load(f)
            all_results.append(user_result)
            continue

        # Step 1: Load user image
        user_img = load_user_image(user_key)

        # Step 2: Get top garment from manifest (pre-computed retrieval)
        manifest = load_manifest()
        if user_key in manifest and manifest[user_key]:
            top_retrieval = manifest[user_key][0]
            garment_filename = top_retrieval["img"]
            garment_category = top_retrieval.get("category_name", "Unknown")
            retrieval_score = top_retrieval.get("score", 0.0)
            print(f"Using manifest retrieval: {garment_filename}, category={garment_category}, score={retrieval_score:.4f}")
        else:
            print(f"WARNING: No manifest entry for {user_key}")
            continue

        garment_path = EXAMPLES_DIR / garment_filename
        if not garment_path.exists():
            print(f"WARNING: Garment image not found: {garment_path}")
            continue

        garment_img = Image.open(garment_path).convert("RGB")

        # Garment index from filename (user_N_garment_K.jpg -> K-1)
        garment_idx = int(garment_filename.split("_")[-1].replace(".jpg", "")) - 1
        print(f"Top garment: idx={garment_idx}, category={garment_category}, score={retrieval_score:.4f}")

        # Step 4: Run VTON
        print(f"Running FasHN-VTO for {user_key}...")
        result_img = None
        fashn_success = False
        fashn_category = top_retrieval.get("fashn_category", "tops")

        if garment_img:
            try:
                result_img = run_tryon(
                    user_img,
                    garment_img,
                    workflow="fashn",
                    category=fashn_category,
                    api_template_path=Path("comfyui_api/FasHN-VTO_api.json"),
                )
                fashn_success = result_img is not None
            except Exception as e:
                print(f"ERROR running VTON: {e}")
                import traceback
                traceback.print_exc()

        # Step 5: Save results
        user_result = {
            "user_key": user_key,
            "garment_idx": garment_idx,
            "garment_category": garment_category,
            "retrieval_score": retrieval_score,
            "sam3_prompt": garment_category.lower().replace("_", ""),
            "vton_success": fashn_success,
            "person_image": f"{user_key}.png",
            "garment_image": garment_filename,
        }

        if result_img:
            result_path = PIPELINE_RESULTS_DIR / f"{user_key}_vton.png"
            result_img.save(result_path)
            user_result["result_image"] = result_path.name
            print(f"Saved: {result_path}")

            # Add metadata to result
            user_result["sample_metadata"] = {
                "garment_idx": garment_idx,
                "garment_category": garment_category,
                "sam3_prompt": garment_category.lower().replace("_", ""),
                "retrieval_score": retrieval_score,
            }

            # Step 6: Evaluate
            if run_evaluation_flag and dino_model:
                print(f"Evaluating {user_key}...")
                sample = {
                    "sample_id": f"{user_key}_pipeline",
                    "user": user_key,
                    "person_path": DEMO_DIR / f"{user_key}.png",
                    "garment_path": garment_path or None,
                    "result_path": result_path,
                    "category_name": garment_category,
                    "fashn_category": fashn_category,
                    "sample_metadata": user_result["sample_metadata"],
                }
                scores = evaluate_sample(
                    sample,
                    dino_model,
                    dino_processor,
                    device,
                    use_vlm=False,
                )
                user_result["metrics"] = scores
                print(f"Metrics: {scores}")

        with open(result_file, "w") as f:
            json.dump(user_result, f, indent=2)

        all_results.append(user_result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="E2E VTON Pipeline")
    parser.add_argument(
        "--users",
        type=str,
        nargs="+",
        default=["user_1"],
        help="User keys to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all demo users",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of garments to retrieve",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip users with existing results",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Copy pipeline results to examples and generate report",
    )
    parser.add_argument(
        "--run-report",
        action="store_true",
        help="Run evaluation report after pipeline",
    )
    args = parser.parse_args()

    user_keys = args.users
    if args.all:
        demo_files = list(DEMO_DIR.glob("*.png"))
        user_keys = [f.stem for f in demo_files]

    if not user_keys:
        print("No users to process")
        return

    results = run_pipeline(
        user_keys,
        top_n=args.top_n,
        skip_existing=args.skip_existing,
        run_evaluation_flag=not args.no_evaluate,
    )

    print(f"\n{'='*50}")
    print(f"Pipeline terminé: {len(results)} résultat(s)")
    print(f"Flag --no-evaluate: {args.no_evaluate}")
    print(f"Flag run_evaluation: {not args.no_evaluate}")
    print(f"Résultats: {PIPELINE_RESULTS_DIR}")

    if args.update_report and results:
        update_examples_from_pipeline(results)

    if args.run_report:
        run_evaluation_report(user_keys)


def update_examples_from_pipeline(results: list[dict]):
    """Copie les résultats du pipeline vers examples et met à jour le manifest."""
    import shutil
    examples_dir = EXAMPLES_DIR
    manifest_path = examples_dir / "manifest.json"

    for r in results:
        user_key = r.get("user_key")
        result_img = r.get("result_image")
        garment_img = r.get("garment_image")

        if result_img and (PIPELINE_RESULTS_DIR / result_img).exists():
            dest = examples_dir / f"{user_key}_garment_1_fashn.png"
            shutil.copy(PIPELINE_RESULTS_DIR / result_img, dest)
            print(f"Copied: {dest.name}")

    print(f"Results copied to: {examples_dir}")


def run_evaluation_report(user_keys: list[str]):
    """Lance l'évaluation et génère le rapport."""
    import subprocess
    print("\nGénération du rapport d'évaluation...")

    cmd = ["uv", "run", "python", "-m", "backend.scripts.evaluate_vton", "--no-vlm", "--samples"]
    cmd.extend(user_keys)

    result = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stderr.decode()[-500:])
    else:
        print(result.stdout.decode()[-500:])
    print("Rapport généré: docs/vto/vto_evaluation.md")


if __name__ == "__main__":
    main()