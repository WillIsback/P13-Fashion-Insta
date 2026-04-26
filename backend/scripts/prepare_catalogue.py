"""
Build data/catalogue.json from the DeepFashion InShop dataset.

Scans dataset/deepfashion-inshop/img_highres for front-view images (*_1_front.jpg),
maps each category to a fashn_category and archetype, and writes data/catalogue.json.

Usage:
    uv run python -m backend.scripts.prepare_catalogue
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASET_ROOT = Path("dataset/deepfashion-inshop/img_highres")
OUT_PATH = Path("data/catalogue.json")

# DeepFashion InShop category → fashn_category
FASHN_MAP = {
    # tops
    "Blouses_Shirts": "tops",
    "Cardigans": "tops",
    "Graphic_Tees": "tops",
    "Jackets_Coats": "tops",
    "Jackets_Vests": "tops",
    "Shirts_Polos": "tops",
    "Suiting": "tops",
    "Sweatshirts_Hoodies": "tops",
    "Sweaters": "tops",
    "Tees_Tanks": "tops",
    # bottoms
    "Denim": "bottoms",
    "Leggings": "bottoms",
    "Pants": "bottoms",
    "Shorts": "bottoms",
    "Skirts": "bottoms",
    # one-pieces
    "Dresses": "one-pieces",
    "Rompers_Jumpsuits": "one-pieces",
}

# category → archetype (based on the distribution in the original catalogue)
ARCHETYPE_MAP = {
    "Blouses_Shirts": "Professionnel moderne",
    "Cardigans": "Essentiel urbain",
    "Denim": "Casual chic",
    "Dresses": "Bohème féminine",
    "Graphic_Tees": "Streetwear",
    "Jackets_Coats": "Streetwear",
    "Jackets_Vests": "Professionnel moderne",
    "Leggings": "Casual chic",
    "Pants": "Professionnel moderne",
    "Rompers_Jumpsuits": "Bohème féminine",
    "Shirts_Polos": "Casual chic",
    "Shorts": "Streetwear",
    "Skirts": "Bohème féminine",
    "Suiting": "Professionnel moderne",
    "Sweatshirts_Hoodies": "Streetwear",
    "Sweaters": "Essentiel urbain",
    "Tees_Tanks": "Casual chic",
}


def build_catalogue() -> list[dict]:
    items = []
    skipped = 0

    for img_path in sorted(DATASET_ROOT.rglob("*_1_front.jpg")):
        # Path structure: img_highres/{GENDER}/{CATEGORY}/{id}/{shot}.jpg
        parts = img_path.parts
        try:
            hires_idx = parts.index("img_highres")
            category = parts[hires_idx + 2]
        except (ValueError, IndexError):
            skipped += 1
            continue

        fashn = FASHN_MAP.get(category)
        archetype = ARCHETYPE_MAP.get(category)
        if fashn is None:
            skipped += 1
            continue

        items.append({
            "filename": img_path.name,
            "path": str(img_path),
            "category_name": category,
            "fashn_category": fashn,
            "archetype": archetype,
            "source": "inshop",
        })

    if skipped:
        print(f"  [warn] Skipped {skipped} images (unknown category or bad path)")

    return items


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {DATASET_ROOT} for front-view images...")
    items = build_catalogue()
    print(f"  Found {len(items):,} catalogue items")

    # Category breakdown
    from collections import Counter
    cats = Counter(i["category_name"] for i in items)
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")

    OUT_PATH.write_text(json.dumps(items, indent=2))
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
