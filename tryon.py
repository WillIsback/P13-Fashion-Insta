"""
ComfyUI virtual try-on client.

Uploads user photo and catalogue item to ComfyUI, injects them into the
Flux2-Klein try-on workflow, submits, polls, and returns the output image.
"""
import copy
import io
import json
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

# Node IDs in the API-format workflow (phase_2/tryon_api.json).
# Verified against actual exported API format.
USER_IMAGE_NODE = "76"
ITEM_IMAGE_NODE = "132"
PROMPT_NODE = "92:113"

COMFYUI_URL = "http://127.0.0.1:8188"
API_TEMPLATE_PATH = Path("phase_2/tryon_api.json")

POLL_INTERVAL = 2.0   # seconds between status checks
POLL_TIMEOUT = 120.0  # seconds before giving up


def inject_params(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    colour_prompt: str,
    user_node_id: str = USER_IMAGE_NODE,
    item_node_id: str = ITEM_IMAGE_NODE,
    prompt_node_id: str = PROMPT_NODE,
) -> dict:
    """
    Return a deep copy of template with user image, item image, and colour prompt injected.
    Does not mutate the input template.
    """
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


def poll_result(prompt_id: str, comfyui_url: str) -> str:
    """
    Poll /history/{prompt_id} until the job completes.
    Returns the output image filename.
    Raises TimeoutError if POLL_TIMEOUT exceeded.
    Raises RuntimeError if ComfyUI reports an error.
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
                        return images[0]["filename"]
                raise RuntimeError("ComfyUI job completed but produced no output images")
        if time.time() < deadline:
            time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"ComfyUI did not complete within {POLL_TIMEOUT}s")


def fetch_output_image(filename: str, comfyui_url: str) -> Image.Image:
    """Fetch the output image from ComfyUI's /view endpoint."""
    resp = requests.get(
        f"{comfyui_url}/view",
        params={"filename": filename, "type": "output"},
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def run_tryon(
    user_img: Image.Image,
    item_img: Image.Image,
    colour: str,
    comfyui_url: str = COMFYUI_URL,
    api_template_path: Path = API_TEMPLATE_PATH,
) -> Image.Image:
    """
    Full try-on pipeline:
      1. Upload user photo and catalogue item to ComfyUI
      2. Inject filenames + colour prompt into workflow
      3. Submit, poll, fetch result
    Returns the output PIL Image.
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

    workflow = inject_params(
        template,
        user_image_name=user_name,
        item_image_name=item_name,
        colour_prompt=colour,
    )

    prompt_id = submit_prompt(workflow, comfyui_url)
    output_filename = poll_result(prompt_id, comfyui_url)
    return fetch_output_image(output_filename, comfyui_url)
