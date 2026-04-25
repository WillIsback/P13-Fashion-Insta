"""
ComfyUI virtual try-on client.

Supports two workflows:
  - "flux"  : Flux2-Klein image editor (phase_2/tryon_api.json)
  - "fashn"  : FasHN-VTO specialized try-on (phase_2/FasHN-VTO_api.json)
"""
import copy
import io
import json
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

# --- Flux2-Klein node IDs (phase_2/tryon_api.json) ---
FLUX_USER_NODE = "76"
FLUX_ITEM_NODE = "132"
FLUX_PROMPT_NODE = "92:113"

# --- FasHN-VTO node IDs (phase_2/FasHN-VTO_api.json) ---
FASHN_PERSON_NODE = "2"
FASHN_GARMENT_NODE = "3"
FASHN_INFERENCE_NODE = "5"
FASHN_CATEGORIES = ("tops", "bottoms", "one-pieces")

COMFYUI_URL = "http://127.0.0.1:8188"
API_TEMPLATE_PATH = Path("phase_2/tryon_api.json")

POLL_INTERVAL = 2.0   # seconds between status checks
POLL_TIMEOUT = 120.0  # seconds before giving up


def inject_params_flux(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    colour_prompt: str,
) -> dict:
    """
    Inject user image, garment image, and colour prompt into Flux2-Klein workflow.
    Returns a deep copy — does not mutate template.
    """
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
    """
    Inject person image, garment image, and garment category into FasHN-VTO workflow.
    category must be one of: 'tops', 'bottoms', 'one-pieces'.
    Returns a deep copy — does not mutate template.
    """
    if category not in FASHN_CATEGORIES:
        raise ValueError(f"category must be one of {FASHN_CATEGORIES}, got {category!r}")
    workflow = copy.deepcopy(template)
    workflow[FASHN_PERSON_NODE]["inputs"]["image"] = user_image_name
    workflow[FASHN_GARMENT_NODE]["inputs"]["image"] = item_image_name
    workflow[FASHN_INFERENCE_NODE]["inputs"]["category"] = category
    return workflow


# Keep old name as alias for backwards compatibility with tests
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
    """
    Upload a PIL image to ComfyUI's input folder via /upload/image.
    Returns the filename assigned by ComfyUI.
    """
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
    """
    Submit workflow (API format) to ComfyUI.
    Returns the prompt_id string.
    """
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"{comfyui_url}/prompt", json=payload)
    response.raise_for_status()
    return response.json()["prompt_id"]


def poll_result(prompt_id: str, comfyui_url: str) -> dict:
    """
    Poll /history/{prompt_id} until the job completes.
    Returns a dict: {"filename": str, "type": str, "subfolder": str}.
    The "type" is "output" for SaveImage nodes and "temp" for PreviewImage nodes.
    Raises TimeoutError if POLL_TIMEOUT exceeded.
    Raises RuntimeError if ComfyUI reports an error or job produces no images.
    """
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(f"{comfyui_url}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()
        if prompt_id in history:
            job = history[prompt_id]
            if job.get("status", {}).get("completed"):
                if "error" in job:
                    raise RuntimeError(f"ComfyUI error: {job['error']}")
                outputs = job.get("outputs", {})
                for node_output in outputs.values():
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
    raise TimeoutError(f"ComfyUI did not complete within {POLL_TIMEOUT}s")


def fetch_output_image(img_info: dict, comfyui_url: str) -> Image.Image:
    """
    Fetch output image from ComfyUI's /view endpoint.
    img_info: dict with keys 'filename', 'type', 'subfolder' (as returned by poll_result).
    """
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
    workflow: str = "flux",
    comfyui_url: str = COMFYUI_URL,
    api_template_path: Path = API_TEMPLATE_PATH,
) -> Image.Image:
    """
    Full try-on pipeline for either workflow.

    workflow="flux"  : Flux2-Klein — uses colour prompt, api_template_path defaults to tryon_api.json
    workflow="fashn" : FasHN-VTO  — uses category (tops/bottoms/one-pieces), ignores colour

    Raises ConnectionError if ComfyUI is unreachable.
    Raises TimeoutError if generation exceeds POLL_TIMEOUT.
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
    else:
        wf = inject_params_flux(template, user_name, item_name, colour or "original colour")

    prompt_id = submit_prompt(wf, comfyui_url)
    img_info = poll_result(prompt_id, comfyui_url)
    return fetch_output_image(img_info, comfyui_url)
