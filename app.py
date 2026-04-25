"""
Gradio UI for the photo-based fashion recommendation PoC.

Run:
    uv run python app.py
"""
from pathlib import Path

import gradio as gr
from PIL import Image

from recommender import recommend
from tryon import run_tryon

EMBEDDINGS_PATH = Path("data/embeddings.npy")
METADATA_PATH = Path("data/index_metadata.json")
DATASET_ROOT = Path("dataset/p13")
TOP_N = 5

SIZES = ["XS", "S", "M", "L", "XL"]

WORKFLOW_FLUX = "phase_2/tryon_api.json"
WORKFLOW_FASHN = "phase_2/FasHN-VTO_api.json"
FASHN_CATEGORIES = ["tops", "bottoms", "one-pieces"]


def _check_index():
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Catalogue index not found. Run: uv run python catalogue.py"
        )


def find_image_path(filename: str) -> Path:
    """Locate a catalogue image file by filename anywhere under DATASET_ROOT."""
    matches = list(DATASET_ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Image {filename} not found in {DATASET_ROOT}")
    return matches[0]


def on_upload(user_photo):
    """Called when user uploads a photo. Returns recommendation gallery data."""
    _check_index()
    if user_photo is None:
        return gr.update(visible=False), [], gr.update(visible=False), []

    img = Image.fromarray(user_photo).convert("RGB")
    results = recommend(img, EMBEDDINGS_PATH, METADATA_PATH, top_n=TOP_N)

    # Build gallery: list of (image_path, label)
    gallery_items = []
    for r in results:
        meta = r["metadata"]
        img_path = find_image_path(meta["fichier"])
        label = f"{meta['productDisplayName']}\n{meta['baseColour']} · {meta['articleType']}\nScore: {r['score']:.2f}"
        gallery_items.append((str(img_path), label))

    return (
        gr.update(visible=True),   # results_section
        gallery_items,             # gallery
        gr.update(visible=False),  # tryon_section
        gallery_items,             # gallery_data state
        gr.update(visible=False),  # error_box — clear on new upload
        gr.update(value=None),     # tryon_result — clear on new upload
    )


def on_select(user_photo, gallery_data: list, evt: gr.SelectData):
    """Called when user clicks a recommendation. Shows customisation panel."""
    if user_photo is None or not gallery_data:
        return gr.update(visible=False), None

    selected_index = evt.index
    # gallery_data is list of (path, label) tuples
    selected_path, selected_label = gallery_data[selected_index]
    selected_img = Image.open(selected_path)

    return (
        gr.update(visible=True),  # show customisation panel
        selected_img,             # preview of selected item
    )


def switch_workflow(current: dict):
    """Toggle between Flux2-Klein and FasHN-VTO workflows."""
    if current["name"] == "Flux2-Klein":
        return (
            {"name": "FasHN-VTO", "path": WORKFLOW_FASHN},
            gr.update(value="Switch to Flux2-Klein", variant="primary"),
            gr.update(value="**Current: FasHN-VTO**"),
            gr.update(visible=False),   # hide colour_input
            gr.update(visible=True),    # show category_input
        )
    else:
        return (
            {"name": "Flux2-Klein", "path": WORKFLOW_FLUX},
            gr.update(value="Switch to FasHN-VTO", variant="secondary"),
            gr.update(value="**Current: Flux2-Klein**"),
            gr.update(visible=True),    # show colour_input
            gr.update(visible=False),   # hide category_input
        )


def on_try_on(user_photo, selected_item_img, colour_text, category, workflow_state: dict):
    """Submit try-on to ComfyUI and return the result image."""
    if user_photo is None or selected_item_img is None:
        return gr.update(value=None, visible=False), gr.update(visible=False)

    user_img = Image.fromarray(user_photo).convert("RGB")
    item_img = (
        selected_item_img.convert("RGB")
        if isinstance(selected_item_img, Image.Image)
        else Image.fromarray(selected_item_img).convert("RGB")
    )

    is_fashn = workflow_state["name"] == "FasHN-VTO"
    api_path = Path(workflow_state["path"])

    try:
        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            colour=colour_text.strip() if colour_text else "",
            category=category or "tops",
            workflow="fashn" if is_fashn else "flux",
            api_template_path=api_path,
        )
    except ConnectionError as e:
        return gr.update(value=None, visible=False), gr.update(value=str(e), visible=True)
    except TimeoutError as e:
        return gr.update(value=None, visible=False), gr.update(value=str(e), visible=True)
    except Exception as e:
        return gr.update(value=None, visible=False), gr.update(value=f"Try-on failed: {e}", visible=True)

    return result, gr.update(visible=False)


def build_ui():
    with gr.Blocks(title="Fashion Recommender PoC") as demo:
        gr.Markdown("# Fashion Recommendation — Photo-Based PoC")

        # --- Upload section ---
        with gr.Row():
            with gr.Column():
                user_photo = gr.Image(label="Upload your photo", type="numpy")
                upload_btn = gr.Button("Find similar items", variant="primary")

        error_box = gr.Textbox(visible=False, label="Error", interactive=False)

        # --- Recommendations section ---
        with gr.Column(visible=False) as results_section:
            gr.Markdown("### Similar items from catalogue")
            gallery = gr.Gallery(
                label="Click an item to select it",
                columns=5,
                height=300,
                allow_preview=False,
            )

        # --- Customisation + try-on section ---
        with gr.Column(visible=False) as tryon_section:
            gr.Markdown("### Selected item")
            with gr.Row():
                selected_preview = gr.Image(label="Selected item", width=200, interactive=False)
                with gr.Column():
                    colour_input = gr.Textbox(
                        label="Colour customisation (e.g. 'in red', 'in navy blue')",
                        placeholder="Leave empty to keep original colour",
                        visible=True,
                    )
                    category_input = gr.Dropdown(
                        choices=FASHN_CATEGORIES,
                        value="tops",
                        label="Garment category (FasHN-VTO only)",
                        visible=False,
                    )
                    size_input = gr.Dropdown(
                        choices=SIZES, value="M", label="Size (display only)"
                    )
                    tryon_btn = gr.Button("Try it on", variant="primary")

            gr.Markdown("### Try-on result")
            tryon_result = gr.Image(label="Virtual try-on", interactive=False)
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                reject_btn = gr.Button("Try another")

        # Internal state
        gallery_data = gr.State([])
        workflow_state = gr.State({"name": "Flux2-Klein", "path": WORKFLOW_FLUX})

        # --- Workflow switch button ---
        with gr.Row():
            workflow_toggle = gr.Button("Switch to FASHN-VTON", variant="secondary")
            workflow_label = gr.Markdown("**Current: Flux2-Klein**")

        # --- Event wiring ---
        upload_btn.click(
            fn=on_upload,
            inputs=[user_photo],
            outputs=[results_section, gallery, tryon_section, gallery_data, error_box, tryon_result],
        )

        gallery.select(
            fn=on_select,
            inputs=[user_photo, gallery_data],
            outputs=[tryon_section, selected_preview],
        )

        tryon_btn.click(
            fn=on_try_on,
            inputs=[user_photo, selected_preview, colour_input, category_input, workflow_state],
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

        # --- Workflow toggle ---
        workflow_toggle.click(
            fn=switch_workflow,
            inputs=[workflow_state],
            outputs=[workflow_state, workflow_toggle, workflow_label, colour_input, category_input],
        )

    return demo


if __name__ == "__main__":
    _check_index()
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
