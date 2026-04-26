"""
ComfyUI virtual try-on client.

Supports three workflows:
  - "flux"  : Flux2-Klein image editor  (phase_2/tryon_api.json)
  - "fashn" : FasHN-VTO specialised try-on (phase_2/FasHN-VTO_api.json)
  - "qwen"  : Qwen-Image-Edit-2511 image editor (phase_2/image_qwen_image_edit_2511_api.json)
"""
import copy
import io
import json
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

# --- Flux2-Klein node IDs ---
FLUX_USER_NODE = "76"
FLUX_ITEM_NODE = "132"
FLUX_PROMPT_NODE = "92:113"

# --- FasHN-VTO node IDs ---
FASHN_PERSON_NODE = "2"
FASHN_GARMENT_NODE = "3"
FASHN_INFERENCE_NODE = "5"
FASHN_CATEGORIES = ("tops", "bottoms", "one-pieces")

# --- Qwen-Image-Edit-2511 node IDs ---
QWEN_PERSON_NODE  = "41"
QWEN_GARMENT_NODE = "83"
QWEN_PROMPT_NODE  = "170:151"

COMFYUI_URL = "http://127.0.0.1:8188"
POLL_INTERVAL = 2.0   # seconds between status checks
POLL_TIMEOUT = 120.0  # seconds before giving up


def inject_params_flux(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    colour_prompt: str,
) -> dict:
    """Inject user image, garment image, and colour prompt into Flux2-Klein workflow."""
    workflow = copy.deepcopy(template)
    workflow[FLUX_USER_NODE]["inputs"]["image"] = user_image_name
    workflow[FLUX_ITEM_NODE]["inputs"]["image"] = item_image_name
    base_prompt = workflow[FLUX_PROMPT_NODE]["inputs"]["text"]
    workflow[FLUX_PROMPT_NODE]["inputs"]["text"] = f"{base_prompt}, {colour_prompt}"
    return workflow


def inject_params_fashn(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    category: str,
) -> dict:
    """Inject person image, garment image, and category into FasHN-VTO workflow."""
    if category not in FASHN_CATEGORIES:
        raise ValueError(f"category must be one of {FASHN_CATEGORIES}, got {category!r}")
    workflow = copy.deepcopy(template)
    workflow[FASHN_PERSON_NODE]["inputs"]["image"] = user_image_name
    workflow[FASHN_GARMENT_NODE]["inputs"]["image"] = item_image_name
    workflow[FASHN_INFERENCE_NODE]["inputs"]["category"] = category
    return workflow


def inject_params_qwen(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    prompt: str,
) -> dict:
    """Inject person image, garment image, and prompt into Qwen-Image-Edit-2511 workflow."""
    workflow = copy.deepcopy(template)
    workflow[QWEN_PERSON_NODE]["inputs"]["image"] = user_image_name
    workflow[QWEN_GARMENT_NODE]["inputs"]["image"] = item_image_name
    workflow[QWEN_PROMPT_NODE]["inputs"]["prompt"] = prompt
    return workflow


def build_qwen_prompt(meta: dict) -> str:
    """Build a preservation-focused try-on prompt from catalogue item metadata."""
    garment = (meta.get("category_name") or "garment").replace("_", " ").lower()
    return (
        f"Virtual try-on: dress the person with the {garment} shown in the reference image. "
        "Preserve the person's face, hair, skin tone, body pose, and the original background "
        "exactly. Only replace the clothing."
    )


def inject_params(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    colour_prompt: str,
    user_node_id: str = FLUX_USER_NODE,
    item_node_id: str = FLUX_ITEM_NODE,
    prompt_node_id: str = FLUX_PROMPT_NODE,
) -> dict:
    """Inject params into Flux2-Klein workflow (backwards-compatible wrapper)."""
    workflow = copy.deepcopy(template)
    workflow[user_node_id]["inputs"]["image"] = user_image_name
    workflow[item_node_id]["inputs"]["image"] = item_image_name
    base_prompt = workflow[prompt_node_id]["inputs"]["text"]
    workflow[prompt_node_id]["inputs"]["text"] = f"{base_prompt}, {colour_prompt}"
    return workflow


def upload_image(img: Image.Image, comfyui_url: str) -> str:
    """Upload a PIL image to ComfyUI. Returns the filename assigned by ComfyUI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    filename = f"poc_{uuid.uuid4().hex[:8]}.png"
    response = requests.post(
        f"{comfyui_url}/upload/image",
        files={"image": (filename, buf, "image/png")},
        data={"overwrite": "true"},
    )
    response.raise_for_status()
    return response.json()["name"]


def submit_prompt(workflow: dict, comfyui_url: str) -> str:
    """Submit workflow to ComfyUI. Returns the prompt_id string."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"{comfyui_url}/prompt", json=payload)
    response.raise_for_status()
    return response.json()["prompt_id"]


def poll_result(prompt_id: str, comfyui_url: str, timeout: float = POLL_TIMEOUT) -> dict:
    """
    Poll until job completes. Returns {"filename", "type", "subfolder"}.
    Raises TimeoutError or RuntimeError on failure.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{comfyui_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            job = history[prompt_id]
            if job.get("status", {}).get("completed"):
                if "error" in job:
                    raise RuntimeError(f"ComfyUI error: {job['error']}")
                for node_output in job.get("outputs", {}).values():
                    images = node_output.get("images", [])
                    if images:
                        img_info = images[0]
                        return {
                            "filename": img_info["filename"],
                            "type": img_info.get("type", "output"),
                            "subfolder": img_info.get("subfolder", ""),
                        }
                raise RuntimeError("ComfyUI job completed but produced no output images")
        if time.time() < deadline:
            time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"ComfyUI did not complete within {timeout}s")


def fetch_output_image(img_info: dict, comfyui_url: str) -> Image.Image:
    """Fetch output image from ComfyUI's /view endpoint."""
    resp = requests.get(
        f"{comfyui_url}/view",
        params={
            "filename": img_info["filename"],
            "type": img_info["type"],
            "subfolder": img_info.get("subfolder", ""),
        },
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def run_tryon(
    user_img: Image.Image,
    item_img: Image.Image,
    colour: str = "",
    category: str = "tops",
    prompt: str = "",
    workflow: str = "flux",
    comfyui_url: str = COMFYUI_URL,
    api_template_path: Path = Path("phase_2/tryon_api.json"),
    poll_timeout: float = POLL_TIMEOUT,
) -> Image.Image:
    """
    Full try-on pipeline.

    workflow="flux"  : Flux2-Klein — uses colour prompt
    workflow="fashn" : FasHN-VTO  — uses category (tops/bottoms/one-pieces)
    workflow="qwen"  : Qwen-Image-Edit-2511 — uses text prompt (auto-built from metadata if empty)

    Raises ConnectionError, TimeoutError, or FileNotFoundError on failure.
    """
    try:
        with open(api_template_path) as f:
            template = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API workflow template not found at {api_template_path}. "
            "Export it from ComfyUI (Dev Mode → Save API Format)."
        )

    try:
        user_name = upload_image(user_img, comfyui_url)
        item_name = upload_image(item_img, comfyui_url)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach ComfyUI at {comfyui_url}. "
            "Make sure the ComfyUI server is running."
        )

    if workflow == "fashn":
        wf = inject_params_fashn(template, user_name, item_name, category)
    elif workflow == "qwen":
        effective_prompt = prompt.strip() if prompt.strip() else build_qwen_prompt({})
        wf = inject_params_qwen(template, user_name, item_name, effective_prompt)
    else:
        wf = inject_params_flux(template, user_name, item_name, colour or "original colour")

    prompt_id = submit_prompt(wf, comfyui_url)
    img_info = poll_result(prompt_id, comfyui_url, timeout=poll_timeout)
    return fetch_output_image(img_info, comfyui_url)
