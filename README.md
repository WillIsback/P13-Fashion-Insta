# Fashion IA — Recommandation photo + Virtual Try-On

PoC de recommandation de vêtements basée sur une photo utilisateur, avec essayage virtuel (VTO) via ComfyUI.

Upload une photo → top-5 vêtements similaires récupérés depuis le catalogue DeepFashion → preview VTO via 3 workflows.

---

## Rapports & Évaluations

### Embedder — Retrieval

| Rapport | Description |
|---------|-------------|
| [Comparaison des embedders](docs/Embedder/retrieval_comparison.md) | Benchmark DINOv3 · FashionCLIP · Marqo FashionSigLIP — qualité visuelle top-5 |
| [Évaluation du recall](docs/Embedder/recall_evaluation.md) | Recall@K sur 6 utilisateurs démo — métriques quantitatives par modèle |

### VTO — Virtual Try-On

| Rapport | Description |
|---------|-------------|
| [Comparaison des modèles VTO](docs/vto/vto_comparison.md) | Flux2-Klein · FasHN-VTO · Qwen-2511 — résultats côte à côte par utilisateur |
| [Évaluation VTON](docs/vto/vto_evaluation.md) | Métriques multi-niveaux : pixel (SSIM/PSNR) · feature (DINOv3) · VLM sémantique (Qwen3-VL) |

---

## Demo

| User | Garment recommandé | Flux2-Klein | FasHN-VTO | Qwen-2511 |
|------|--------------------|:-----------:|:---------:|:---------:|
| ![user1](docs/vto/examples/user_1.png) | ![g1](docs/vto/examples/user_1_garment_1.jpg) | ![flux](docs/vto/examples/user_1_garment_1_flux.png) | ![fashn](docs/vto/examples/user_1_garment_1_fashn.png) | ![qwen](docs/vto/examples/user_1_garment_1_qwen.png) |

Rapport complet : [`docs/vto/vto_comparison.md`](docs/vto/vto_comparison.md)

---

## Architecture

```
Photo utilisateur
       │
       ▼
  background removal (rembg)
       │
       ▼
  Marqo FashionSigLIP embedding (768-dim)
       │
       ▼
  cosine similarity sur le catalogue DeepFashion InShop
       │
       ▼
  Top-5 garments
       │
       ▼
  ComfyUI VTO  ──┬── Flux2-Klein (FLUX.1 inpainting)
                 ├── FasHN-VTO   (modèle try-on dédié)
                 └── Qwen-Image-Edit-2511 (instruction-following)
```

**Stack :** Python 3.12 · Gradio · PyTorch · open-clip · rembg · YOLOv8 · ComfyUI

---

## Prérequis

### Système

| Composant | Requis | Recommandé |
|-----------|--------|------------|
| Python | 3.12+ | 3.12 |
| GPU | non (CPU possible) | NVIDIA CUDA 12+ (16 GB VRAM pour VTO) |
| RAM | 8 GB | 32 GB |
| Stockage | ~15 GB (dataset + data) | SSD |
| ComfyUI | oui, pour la partie VTO | avec les modèles Flux2-Klein, FasHN, Qwen |

### Outils Python

```bash
# uv (gestionnaire de paquets rapide)
curl -Ls https://astral.sh/uv/install.sh | sh

# Cloner le dépôt
git clone <repo-url>
cd P13
```

---

## Installation

```bash
# Créer l'environnement et installer les dépendances
uv sync

# Copier le fichier d'environnement et remplir les tokens
cp .env.example .env
```

Éditer `.env` :

```
KAGGLE_USERNAME=votre_username_kaggle
KAGGLE_API_TOKEN=votre_token_kaggle
HF_TOKEN=votre_token_huggingface   # optionnel
```

---

## Données

### 1. DeepFashion InShop (catalogue de vêtements)

Le dataset officiel est disponible sur Kaggle :

**Option A — Kaggle CLI** (recommandé)

```bash
# Nécessite KAGGLE_USERNAME + KAGGLE_API_TOKEN dans .env
source .env

kaggle datasets download hserdaraltan/deepfashion-inshop-clothes-retrieval \
    -p dataset/deepfashion-inshop --unzip
```

**Option B — kagglehub (Python)**

```python
import kagglehub

path = kagglehub.dataset_download("hserdaraltan/deepfashion-inshop-clothes-retrieval")
print("Path to dataset files:", path)
```

**Option C — cURL**

```bash
curl -L -o ~/Downloads/deepfashion-inshop-clothes-retrieval.zip \
  https://www.kaggle.com/api/v1/datasets/download/hserdaraltan/deepfashion-inshop-clothes-retrieval
# Puis décompresser dans dataset/deepfashion-inshop/
```

> Dataset Kaggle : [`hserdaraltan/deepfashion-inshop-clothes-retrieval`](https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval)

Structure attendue après décompression :

```
dataset/
└── deepfashion-inshop/
    └── img_highres/
        ├── MEN/
        │   ├── Denim/
        │   │   └── id_000XXXXX/
        │   │       ├── 01_1_front.jpg
        │   │       └── ...
        │   └── ...
        └── WOMEN/
            └── ...
```

### 2. Modèle YOLOv8n (segmentation)

```bash
# Téléchargement automatique via uv
uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Ou manuellement depuis [Ultralytics releases](https://github.com/ultralytics/assets/releases) → placer `yolov8n.pt` à la racine du projet.

---

## Construire les index

Ces étapes sont à exécuter une seule fois après le téléchargement des données.

```bash
# 1. Générer le catalogue (data/catalogue.json)
uv run python -m backend.scripts.prepare_catalogue

# 2. Construire l'index DINOv3 ViT-H (data/embeddings.npy)
uv run python -m backend.scripts.build_index

# 3. Construire les index CLIP (FashionCLIP + Marqo FashionSigLIP)
#    → data/embeddings_fashion_clip.npy
#    → data/embeddings_marqo_fashion_siglip.npy
uv run python -m backend.scripts.build_clip_indices
```

> La construction complète nécessite un GPU. Sur CPU seul, prévoir plusieurs heures.
> Les scripts supportent le **checkpoint resume** : interruptibles et relançables.

---

## Lancer l'application

```bash
# ComfyUI doit tourner sur http://127.0.0.1:8188 pour la partie VTO
# (facultatif si vous n'utilisez pas le try-on)

uv run python main.py
```

L'interface Gradio est accessible sur : **http://127.0.0.1:7860**

---

## Workflows ComfyUI

Les fichiers workflow sont dans `comfyui_api/` :

| Fichier | Modèle | Usage |
|---------|--------|-------|
| `FasHN-VTO_api.json` | FasHN-VTO | Try-on dédié (person + garment + category) |
| `image_flux2_klein_image_edit_4b_base_api.json` | Flux2-Klein (FLUX.1 4B) | Image editing par inpainting |
| `image_qwen_image_edit_2511_api.json` | Qwen-VL 7B | Instruction-following image edit |

Les modèles ComfyUI correspondants doivent être installés dans votre instance ComfyUI (`models/` et custom nodes).

---

## Tests

```bash
uv run pytest tests/ -v
```

---

## Structure du projet

```
P13/
├── backend/
│   ├── core/
│   │   ├── embedder.py       # Marqo FashionSigLIP + DINOv3 embeddings
│   │   ├── retrieval.py      # cosine similarity search
│   │   └── tryon.py          # client ComfyUI (3 workflows)
│   └── scripts/
│       ├── prepare_catalogue.py   # scan dataset → data/catalogue.json
│       ├── build_index.py         # DINOv3 embeddings
│       ├── build_clip_indices.py  # CLIP embeddings
│       ├── generate_retrieval_report.py
│       └── generate_vto_report.py
├── comfyui_api/          # workflows ComfyUI (JSON)
├── demo/                 # photos utilisateurs de démo
├── docs/                 # rapports, specs, comparaisons VTO
├── frontend/
│   └── ui.py             # interface Gradio
├── tests/
├── main.py               # point d'entrée
├── pyproject.toml
└── .env.example
```

---

## Références & Third Parties

### Modèles utilisés

| Modèle | Usage | Lien |
|--------|-------|------|
| **Marqo FashionSigLIP** | Embedding image mode retrieval (768-dim) | [HuggingFace](https://huggingface.co/Marqo/marqo-fashionSigLIP) |
| **DINOv3 ViT-H+/16** | Feature extraction — évaluation identité & fidélité vêtement | [HuggingFace](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) |
| **FashionCLIP** | Index CLIP fashion alternatif | [HuggingFace](https://huggingface.co/patrickjohncyh/fashion-clip) |
| **FasHN-VTO-1.5** | Virtual try-on dédié (person + garment + category) | [HuggingFace](https://huggingface.co/fashn-ai/fashn-vton-1.5) |
| **FLUX.1 / Flux2-Klein 4B** | Image editing par inpainting (workflow ComfyUI) | [Black Forest Labs](https://blackforestlabs.ai) |
| **Qwen-Image-Edit-2511 7B** | Instruction-following image editing (workflow ComfyUI) | [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| **Qwen3-VL-4B-Instruct** | Scoring VLM sémantique (évaluation VTON 5 dimensions) | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |

### Outils & Frameworks

| Outil | Usage | Lien |
|-------|-------|------|
| **ComfyUI** | Moteur d'inférence VTO (graph-based, API JSON) | [GitHub](https://github.com/comfyanonymous/ComfyUI) |
| **rembg** | Suppression de fond (u2net) — isolation vêtement | [GitHub](https://github.com/danielgatis/rembg) |
| **OpenVTON-Bench** | Protocole d'évaluation VTON (SAM3 + DINOv3 + métriques multi-échelles) | [GitHub](https://github.com/RenxingIntelligence/OpenVTON-Bench) |
| **DeepFashion InShop** | Dataset catalogue vêtements (>52k images, retrieval benchmark) | [Kaggle](https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval) |
| **Gradio** | Interface utilisateur web | [gradio.app](https://www.gradio.app) |

### Papers — Virtual Try-On (état de l'art étudié)

| Papier | Modèle | arXiv |
|--------|--------|-------|
| *Transformer-Empowered Diffusion for Virtual Try-On* | TED-VITON | [2411.17017](https://arxiv.org/abs/2411.17017) |
| *VTON-360: High-Fidelity Virtual Try-On from Any Viewing Angle* | VTON-360 | [2503.12165](https://arxiv.org/abs/2503.12165) |
| *OmniTry: Flexible Virtual Try-On via Unified Paradigm* | OmniTry | [2508.13632](https://arxiv.org/abs/2508.13632) |
| *OmniVTON: Unified Garment-Centric Virtual Try-On* | OmniVTON | [2507.15037](https://arxiv.org/abs/2507.15037) |
| *OmniVTON++* | OmniVTON++ | [2602.14552](https://arxiv.org/abs/2602.14552) |
| *PROMO: Prompt-based Outfit Recommendation* | PROMO | [2603.11675](https://arxiv.org/abs/2603.11675) |
| *SMF-VTO: Structure-aware Multi-flow Virtual Try-On* | SMF-VTO | [2603.29587](https://arxiv.org/abs/2603.29587) |
| *DEFT-VTON: Deformable Feature Transfer for VTON* | DEFT-VTON | [2509.13506](https://arxiv.org/abs/2509.13506) |
| *DiT-VTON: Diffusion Transformer for Virtual Try-On* | DiT-VTON | [2510.04797](https://arxiv.org/abs/2510.04797) |
| *TEMU-VTOFF: Try-On and Try-Off Unified Framework* | TEMU-VTOFF | [2505.21062](https://arxiv.org/abs/2505.21062) |
| *Tstars-Tryon-1.0* | Tstars | [2604.19748](https://arxiv.org/abs/2604.19748) |
| *FIT Dataset: Fashion Image Try-on Benchmark* | FIT | [2604.08526](https://arxiv.org/abs/2604.08526) |
