"""
Gradio UI for the photo-based fashion recommendation PoC.

Imported by main.py — do not run directly.
"""
from pathlib import Path

import gradio as gr
from PIL import Image

from backend.core.retrieval import recommend
from backend.core.tryon import build_qwen_prompt, run_tryon

EMBEDDINGS_PATH = Path("data/embeddings.npy")
METADATA_PATH = Path("data/index_metadata.json")
TOP_N = 5

SIZES = ["XS", "S", "M", "L", "XL"]

WORKFLOW_FLUX = "phase_2/tryon_api.json"
WORKFLOW_FASHN = "phase_2/FasHN-VTO_api.json"
WORKFLOW_QWEN = "phase_2/image_qwen_image_edit_2511_api.json"
FASHN_CATEGORIES = ["tops", "bottoms", "one-pieces"]


def _check_index():
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Catalogue index not found. Run: uv run python -m backend.scripts.build_index"
        )


def find_image_path(meta: dict) -> Path:
    """Return the catalogue image path from metadata."""
    p = Path(meta["path"])
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def on_upload(user_photo):
    """Called when user uploads a photo. Returns recommendation gallery data."""
    _check_index()
    if user_photo is None:
        return gr.update(visible=False), [], gr.update(visible=False), [], gr.update(visible=False), gr.update(value=None), []

    img = user_photo.convert("RGB")
    results = recommend(img, EMBEDDINGS_PATH, METADATA_PATH, top_n=TOP_N)

    gallery_items = []
    for r in results:
        meta = r["metadata"]
        img_path = find_image_path(meta)
        label = (
            f"{meta['category_name']} · {meta['fashn_category']}\n"
            f"{meta['archetype']}\nScore: {r['score']:.2f}"
        )
        gallery_items.append((str(img_path), label))

    return (
        gr.update(visible=True),
        gallery_items,
        gr.update(visible=False),
        gallery_items,
        gr.update(visible=False),
        gr.update(value=None),
        results,
    )


def on_select(user_photo, gallery_data: list, catalogue_results: list, evt: gr.SelectData):
    """Called when user clicks a recommendation. Shows customisation panel."""
    if user_photo is None or not gallery_data:
        return gr.update(visible=False), None, gr.update()

    selected_path, _ = gallery_data[evt.index]
    meta = catalogue_results[evt.index]["metadata"] if catalogue_results else {}
    qwen_prompt = build_qwen_prompt(meta)
    return gr.update(visible=True), Image.open(selected_path), gr.update(value=qwen_prompt)


def switch_workflow(current: dict):
    """Cycle through Flux2-Klein → FasHN-VTO → Qwen → Flux2-Klein."""
    if current["name"] == "Flux2-Klein":
        return (
            {"name": "FasHN-VTO", "path": WORKFLOW_FASHN},
            gr.update(value="Switch to Qwen", variant="primary"),
            gr.update(value="**Current: FasHN-VTO**"),
            gr.update(visible=False),   # colour_input
            gr.update(visible=True),    # category_input
            gr.update(visible=False),   # prompt_input
        )
    elif current["name"] == "FasHN-VTO":
        return (
            {"name": "Qwen", "path": WORKFLOW_QWEN},
            gr.update(value="Switch to Flux2-Klein", variant="secondary"),
            gr.update(value="**Current: Qwen-Image-Edit-2511**"),
            gr.update(visible=False),   # colour_input
            gr.update(visible=False),   # category_input
            gr.update(visible=True),    # prompt_input
        )
    else:  # Qwen
        return (
            {"name": "Flux2-Klein", "path": WORKFLOW_FLUX},
            gr.update(value="Switch to FasHN-VTO", variant="secondary"),
            gr.update(value="**Current: Flux2-Klein**"),
            gr.update(visible=True),    # colour_input
            gr.update(visible=False),   # category_input
            gr.update(visible=False),   # prompt_input
        )


def on_try_on(user_photo, selected_item_img, colour_text, category, prompt_text, workflow_state: dict):
    """Submit try-on to ComfyUI and return the result image."""
    if user_photo is None or selected_item_img is None:
        return gr.update(value=None, visible=False), gr.update(visible=False)

    user_img = user_photo.convert("RGB")
    item_img = selected_item_img.convert("RGB")

    name = workflow_state["name"]
    api_path = Path(workflow_state["path"])

    try:
        if name == "FasHN-VTO":
            result = run_tryon(
                user_img=user_img,
                item_img=item_img,
                category=category or "tops",
                workflow="fashn",
                api_template_path=api_path,
            )
        elif name == "Qwen":
            result = run_tryon(
                user_img=user_img,
                item_img=item_img,
                prompt=prompt_text.strip() if prompt_text else "",
                workflow="qwen",
                api_template_path=api_path,
            )
        else:  # Flux2-Klein
            result = run_tryon(
                user_img=user_img,
                item_img=item_img,
                colour=colour_text.strip() if colour_text else "",
                workflow="flux",
                api_template_path=api_path,
            )
    except ConnectionError as e:
        return gr.update(value=None, visible=False), gr.update(value=str(e), visible=True)
    except TimeoutError as e:
        return gr.update(value=None, visible=False), gr.update(value=str(e), visible=True)
    except Exception as e:
        return gr.update(value=None, visible=False), gr.update(value=f"Try-on failed: {e}", visible=True)

    return result, gr.update(visible=False)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Fashion Recommender PoC") as demo:
        gr.Markdown("# Fashion Recommendation — Photo-Based PoC")

        with gr.Row():
            with gr.Column():
                user_photo = gr.Image(label="Upload your photo", type="pil")
                upload_btn = gr.Button("Find similar items", variant="primary")

        error_box = gr.Textbox(visible=False, label="Error", interactive=False)

        with gr.Column(visible=False) as results_section:
            gr.Markdown("### Similar items from catalogue")
            gallery = gr.Gallery(
                label="Click an item to select it",
                columns=5,
                height=300,
                allow_preview=False,
            )

        with gr.Column(visible=False) as tryon_section:
            gr.Markdown("### Selected item")
            with gr.Row():
                selected_preview = gr.Image(label="Selected item", width=200, interactive=False, type="pil")
                with gr.Column():
                    colour_input = gr.Textbox(
                        label="Colour customisation (e.g. 'in red')",
                        placeholder="Leave empty to keep original colour",
                        visible=True,
                    )
                    category_input = gr.Dropdown(
                        choices=FASHN_CATEGORIES,
                        value="tops",
                        label="Garment category (FasHN-VTO only)",
                        visible=False,
                    )
                    prompt_input = gr.Textbox(
                        label="Try-on prompt (Qwen)",
                        placeholder="Auto-filled from garment annotations — edit freely",
                        visible=False,
                    )
                    size_input = gr.Dropdown(choices=SIZES, value="M", label="Size (display only)")
                    tryon_btn = gr.Button("Try it on", variant="primary")

            gr.Markdown("### Try-on result")
            tryon_result = gr.Image(label="Virtual try-on", interactive=False, type="pil")
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                reject_btn = gr.Button("Try another")

        gallery_data = gr.State([])
        catalogue_results = gr.State([])
        workflow_state = gr.State({"name": "Flux2-Klein", "path": WORKFLOW_FLUX})

        with gr.Row():
            workflow_toggle = gr.Button("Switch to FasHN-VTO", variant="secondary")
            workflow_label = gr.Markdown("**Current: Flux2-Klein**")

        upload_btn.click(
            fn=on_upload,
            inputs=[user_photo],
            outputs=[results_section, gallery, tryon_section, gallery_data, error_box, tryon_result, catalogue_results],
        )

        gallery.select(
            fn=on_select,
            inputs=[user_photo, gallery_data, catalogue_results],
            outputs=[tryon_section, selected_preview, prompt_input],
        )

        tryon_btn.click(
            fn=on_try_on,
            inputs=[user_photo, selected_preview, colour_input, category_input, prompt_input, workflow_state],
            outputs=[tryon_result, error_box],
        )

        approve_btn.click(
            fn=lambda: (None, None, gr.update(visible=False), gr.update(visible=False)),
            outputs=[tryon_result, selected_preview, tryon_section, results_section],
        )

        reject_btn.click(
            fn=lambda: (None, None, gr.update(visible=False)),
            outputs=[tryon_result, selected_preview, tryon_section],
        )

        workflow_toggle.click(
            fn=switch_workflow,
            inputs=[workflow_state],
            outputs=[workflow_state, workflow_toggle, workflow_label, colour_input, category_input, prompt_input],
        )

    return demo
