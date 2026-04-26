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

WORKFLOW_FLUX = "comfyui_api/image_flux2_klein_image_edit_4b_base_api.json"
WORKFLOW_FASHN = "comfyui_api/FasHN-VTO_api.json"
WORKFLOW_QWEN = "comfyui_api/image_qwen_image_edit_2511_api.json"
FASHN_CATEGORIES = ["tops", "bottoms", "one-pieces"]

WORKFLOW_OPTIONS = ["Flux2-Klein", "FasHN-VTO", "Qwen-Image-Edit-2511"]

WORKFLOW_META = {
    "Flux2-Klein":          {"key": "flux",  "path": WORKFLOW_FLUX},
    "FasHN-VTO":            {"key": "fashn", "path": WORKFLOW_FASHN},
    "Qwen-Image-Edit-2511": {"key": "qwen",  "path": WORKFLOW_QWEN},
}

CSS = """
/* ── Global ── */
.gradio-container { max-width: 1280px !important; margin: 0 auto; }

/* ── Upload card ── */
#upload-card {
    background: var(--background-fill-secondary);
    border-radius: var(--radius-lg);
    padding: 16px;
}

/* ── Gallery ── */
#results-gallery .thumbnail-item { cursor: pointer; }
#results-gallery .thumbnail-item img {
    transition: transform 0.15s ease;
}
#results-gallery .thumbnail-item:hover img { transform: scale(1.04); }

/* ── Try-on panel ── */
#tryon-panel {
    border-top: 1px solid var(--border-color-primary);
    padding-top: 16px;
    margin-top: 8px;
}

/* ── Result image ── */
#tryon-result img {
    border-radius: var(--radius-lg);
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}

/* ── Selected preview ── */
#selected-preview img {
    border-radius: var(--radius-md);
    border: 2px solid var(--border-color-accent);
}

/* ── Workflow radio ── */
#workflow-radio .wrap { gap: 8px; }
#workflow-radio label { cursor: pointer; }

/* ── Error box ── */
#error-box textarea {
    color: var(--error-text-color, #b91c1c);
    background: #fef2f2;
    border-color: #fca5a5;
    font-size: 0.875rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: var(--primary-200);
    color: var(--primary-700);
}
"""


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
            f"{meta['archetype']}  —  {r['score']:.2f}"
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
    meta = catalogue_results[evt.index]["metadata"] if len(catalogue_results) > evt.index else {}
    qwen_prompt = build_qwen_prompt(meta)
    return gr.update(visible=True), Image.open(selected_path), gr.update(value=qwen_prompt)


def on_workflow_change(workflow_name: str):
    """Show/hide the correct input control for the selected workflow."""
    return (
        gr.update(visible=(workflow_name == "Flux2-Klein")),
        gr.update(visible=(workflow_name == "FasHN-VTO")),
        gr.update(visible=(workflow_name == "Qwen-Image-Edit-2511")),
    )


def on_try_on(user_photo, selected_item_img, colour_text, category, prompt_text, workflow_name: str):
    """Submit try-on to ComfyUI and return the result image."""
    if user_photo is None or selected_item_img is None:
        return gr.update(value=None, visible=False), gr.update(visible=False)

    user_img = user_photo.convert("RGB")
    item_img = selected_item_img.convert("RGB")

    meta = WORKFLOW_META[workflow_name]
    api_path = Path(meta["path"])

    try:
        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            colour=colour_text.strip() if colour_text else "",
            category=category or "tops",
            prompt=prompt_text.strip() if prompt_text else "",
            workflow=meta["key"],
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
    with gr.Blocks(
        title="Fashion Recommender PoC",
        theme=gr.themes.Soft(primary_hue="stone", neutral_hue="slate"),
        css=CSS,
        fill_width=False,
    ) as demo:

        gr.Markdown(
            "# Fashion Recommender\n"
            "Upload a photo → pick a garment → try it on with any of the 3 VTO models."
        )

        # ── Row 1: Upload + Gallery ──────────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260, elem_id="upload-card"):
                gr.Markdown("### Your photo")
                user_photo = gr.Image(
                    label=None,
                    type="pil",
                    show_label=False,
                    height=340,
                )
                upload_btn = gr.Button("Find similar items", variant="primary", size="lg")

            with gr.Column(scale=3):
                with gr.Column(visible=False) as results_section:
                    gr.Markdown("### Similar items — click one to try it on")
                    gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        elem_id="results-gallery",
                        columns=5,
                        rows=1,
                        height=320,
                        object_fit="contain",
                        allow_preview=False,
                    )

        error_box = gr.Textbox(
            visible=False,
            label="Error",
            interactive=False,
            elem_id="error-box",
        )

        # ── Row 2: Try-on panel ───────────────────────────────────────────────
        with gr.Column(visible=False, elem_id="tryon-panel") as tryon_section:
            with gr.Row(equal_height=True):

                # Selected item preview + controls
                with gr.Column(scale=1, min_width=220):
                    gr.Markdown("#### Selected garment")
                    selected_preview = gr.Image(
                        label=None,
                        show_label=False,
                        elem_id="selected-preview",
                        interactive=False,
                        type="pil",
                        height=280,
                    )
                    size_input = gr.Dropdown(
                        choices=SIZES,
                        value="M",
                        label="Size (display only)",
                    )

                # Workflow + controls
                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("#### VTO model")
                    workflow_radio = gr.Radio(
                        choices=WORKFLOW_OPTIONS,
                        value="Flux2-Klein",
                        label=None,
                        show_label=False,
                        elem_id="workflow-radio",
                    )
                    colour_input = gr.Textbox(
                        label="Colour customisation",
                        placeholder="e.g. 'in red' — leave empty to keep original",
                        visible=True,
                    )
                    category_input = gr.Dropdown(
                        choices=FASHN_CATEGORIES,
                        value="tops",
                        label="Garment category",
                        visible=False,
                    )
                    prompt_input = gr.Textbox(
                        label="Try-on prompt",
                        placeholder="Auto-filled from annotations — edit freely",
                        lines=3,
                        visible=False,
                    )
                    tryon_btn = gr.Button("Try it on ✦", variant="primary", size="lg")

                # Result
                with gr.Column(scale=2, min_width=320):
                    gr.Markdown("#### Result")
                    tryon_result = gr.Image(
                        label=None,
                        show_label=False,
                        elem_id="tryon-result",
                        interactive=False,
                        type="pil",
                        height=340,
                    )
                    with gr.Row():
                        approve_btn = gr.Button("✓ Approve", variant="primary")
                        reject_btn = gr.Button("✗ Try another", variant="secondary")

        # ── State ────────────────────────────────────────────────────────────
        gallery_data = gr.State([])
        catalogue_results = gr.State([])

        # ── Event wiring ─────────────────────────────────────────────────────
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

        workflow_radio.change(
            fn=on_workflow_change,
            inputs=[workflow_radio],
            outputs=[colour_input, category_input, prompt_input],
        )

        tryon_btn.click(
            fn=on_try_on,
            inputs=[user_photo, selected_preview, colour_input, category_input, prompt_input, workflow_radio],
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

    return demo
