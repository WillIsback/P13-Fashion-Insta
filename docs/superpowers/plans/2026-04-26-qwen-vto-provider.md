# Qwen-Image-Edit-2511 VTO Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen-Image-Edit-2511 as a third virtual try-on provider, with annotation-driven prompt auto-fill and a 3-state workflow toggle in the UI.

**Architecture:** Two files change — `backend/core/tryon.py` gains a new inject function, a prompt builder, and a `run_tryon` branch; `frontend/ui.py` gains a `catalogue_results` state, a `prompt_input` textbox, and extends the toggle to a 3-state cycle. No new modules, no new dependencies.

**Tech Stack:** Python 3.12, PIL, Gradio, requests, pytest, unittest.mock

---

## File Map

| File | Change |
|------|--------|
| `backend/core/tryon.py` | Add constants, `inject_params_qwen`, `build_qwen_prompt`, extend `run_tryon` |
| `frontend/ui.py` | Add `catalogue_results` state, `prompt_input` widget, extend toggle + wiring |
| `tests/test_tryon.py` | Add 4 new test functions for Qwen backend |

---

### Task 1: Backend — constants, inject function, prompt builder

**Files:**
- Modify: `backend/core/tryon.py`
- Test: `tests/test_tryon.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tryon.py`:

```python
def make_qwen_api_template():
    return {
        "41": {"class_type": "LoadImage", "inputs": {"image": "old_person.png"}},
        "83": {"class_type": "LoadImage", "inputs": {"image": "old_garment.jpg"}},
        "170:151": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": "old prompt", "clip": ["170:162", 0], "vae": ["170:146", 0],
                       "image1": ["170:160", 0], "image2": ["83", 0]},
        },
    }


FAKE_QWEN_INFO = {"filename": "Qwen_Edit_2511_00001_.png", "type": "output", "subfolder": ""}


def test_inject_params_qwen_modifies_correct_nodes():
    from backend.core.tryon import inject_params_qwen

    template = make_qwen_api_template()
    result = inject_params_qwen(template, "user_qwen.png", "garment_qwen.jpg", "try on the jacket")

    assert result["41"]["inputs"]["image"] == "user_qwen.png"
    assert result["83"]["inputs"]["image"] == "garment_qwen.jpg"
    assert result["170:151"]["inputs"]["prompt"] == "try on the jacket"


def test_inject_params_qwen_does_not_mutate_template():
    from backend.core.tryon import inject_params_qwen

    template = make_qwen_api_template()
    original_person = template["41"]["inputs"]["image"]
    original_prompt = template["170:151"]["inputs"]["prompt"]
    inject_params_qwen(template, "new_user.png", "new_garment.jpg", "new prompt")
    assert template["41"]["inputs"]["image"] == original_person
    assert template["170:151"]["inputs"]["prompt"] == original_prompt


def test_build_qwen_prompt_uses_category_name():
    from backend.core.tryon import build_qwen_prompt

    meta = {"category_name": "Graphic_Tees", "fashn_category": "tops", "archetype": "Streetwear"}
    prompt = build_qwen_prompt(meta)
    assert "graphic tees" in prompt.lower()
    assert "preserve" in prompt.lower() or "keep" in prompt.lower()


def test_build_qwen_prompt_fallback_on_missing_meta():
    from backend.core.tryon import build_qwen_prompt

    prompt = build_qwen_prompt({})
    assert isinstance(prompt, str)
    assert len(prompt) > 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tryon.py::test_inject_params_qwen_modifies_correct_nodes tests/test_tryon.py::test_inject_params_qwen_does_not_mutate_template tests/test_tryon.py::test_build_qwen_prompt_uses_category_name tests/test_tryon.py::test_build_qwen_prompt_fallback_on_missing_meta -v
```

Expected: 4 FAILs with `ImportError: cannot import name 'inject_params_qwen'`

- [ ] **Step 3: Add constants, `inject_params_qwen`, and `build_qwen_prompt` to `backend/core/tryon.py`**

After the `FASHN_CATEGORIES` line (line 27), add:

```python
# --- Qwen-Image-Edit-2511 node IDs ---
QWEN_PERSON_NODE  = "41"
QWEN_GARMENT_NODE = "83"
QWEN_PROMPT_NODE  = "170:151"
```

After the `inject_params_fashn` function, add:

```python
def inject_params_qwen(
    template: dict,
    user_image_name: str,
    item_image_name: str,
    prompt: str,
) -> dict:
    """Inject person image, garment image, and prompt into Qwen-Image-Edit-2511 workflow."""
    workflow = copy.deepcopy(template)
    workflow[QWEN_PERSON_NODE]["inputs"]["image"]  = user_image_name
    workflow[QWEN_GARMENT_NODE]["inputs"]["image"] = item_image_name
    workflow[QWEN_PROMPT_NODE]["inputs"]["prompt"] = prompt
    return workflow


def build_qwen_prompt(meta: dict) -> str:
    """Build a preservation-focused try-on prompt from catalogue item metadata."""
    garment = meta.get("category_name", "garment").replace("_", " ").lower()
    return (
        f"Virtual try-on: dress the person with the {garment} shown in the reference image. "
        "Preserve the person's face, hair, skin tone, body pose, and the original background "
        "exactly. Only replace the clothing."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tryon.py::test_inject_params_qwen_modifies_correct_nodes tests/test_tryon.py::test_inject_params_qwen_does_not_mutate_template tests/test_tryon.py::test_build_qwen_prompt_uses_category_name tests/test_tryon.py::test_build_qwen_prompt_fallback_on_missing_meta -v
```

Expected: 4 PASSes

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
uv run pytest tests/test_tryon.py -v
```

Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add backend/core/tryon.py tests/test_tryon.py
git commit -m "feat: add Qwen-Image-Edit-2511 inject function and prompt builder"
```

---

### Task 2: Backend — extend `run_tryon` for Qwen

**Files:**
- Modify: `backend/core/tryon.py`
- Test: `tests/test_tryon.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tryon.py`:

```python
def test_run_tryon_qwen_returns_pil_image(tmp_path):
    from backend.core.tryon import run_tryon

    api_template_path = tmp_path / "qwen_api.json"
    api_template_path.write_text(json.dumps(make_qwen_api_template()))

    user_img = Image.new("RGB", (64, 64))
    item_img = Image.new("RGB", (64, 64))

    with patch("backend.core.tryon.upload_image", side_effect=["user.png", "item.png"]), \
         patch("backend.core.tryon.submit_prompt", return_value="pid-qwen") as mock_submit, \
         patch("backend.core.tryon.poll_result", return_value=FAKE_QWEN_INFO), \
         patch("backend.core.tryon.fetch_output_image", return_value=Image.new("RGB", (512, 512))):

        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            prompt="Virtual try-on: dress the person with the dress shown in the reference image.",
            workflow="qwen",
            comfyui_url="http://127.0.0.1:8188",
            api_template_path=api_template_path,
        )

    assert isinstance(result, Image.Image)
    submitted_workflow = mock_submit.call_args[0][0]
    assert submitted_workflow["41"]["inputs"]["image"] == "user.png"
    assert submitted_workflow["83"]["inputs"]["image"] == "item.png"
    assert "dress" in submitted_workflow["170:151"]["inputs"]["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_tryon.py::test_run_tryon_qwen_returns_pil_image -v
```

Expected: FAIL — `run_tryon` doesn't handle `workflow="qwen"` yet

- [ ] **Step 3: Extend `run_tryon` in `backend/core/tryon.py`**

Add `prompt: str = ""` to the `run_tryon` signature:

```python
def run_tryon(
    user_img: Image.Image,
    item_img: Image.Image,
    colour: str = "",
    category: str = "tops",
    prompt: str = "",
    workflow: str = "flux",
    comfyui_url: str = COMFYUI_URL,
    api_template_path: Path = Path("phase_2/tryon_api.json"),
) -> Image.Image:
```

Replace the existing `if workflow == "fashn":` block (currently at the bottom before `submit_prompt`) with:

```python
    if workflow == "fashn":
        wf = inject_params_fashn(template, user_name, item_name, category)
    elif workflow == "qwen":
        effective_prompt = prompt.strip() if prompt.strip() else build_qwen_prompt({})
        wf = inject_params_qwen(template, user_name, item_name, effective_prompt)
    else:
        wf = inject_params_flux(template, user_name, item_name, colour or "original colour")
```

Also update the docstring to document the new workflow:

```python
    """
    Full try-on pipeline.

    workflow="flux"  : Flux2-Klein — uses colour prompt
    workflow="fashn" : FasHN-VTO  — uses category (tops/bottoms/one-pieces)
    workflow="qwen"  : Qwen-Image-Edit-2511 — uses text prompt (auto-built from metadata if empty)

    Raises ConnectionError, TimeoutError, or FileNotFoundError on failure.
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_tryon.py::test_run_tryon_qwen_returns_pil_image -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/test_tryon.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/core/tryon.py tests/test_tryon.py
git commit -m "feat: extend run_tryon with qwen workflow branch"
```

---

### Task 3: Frontend — state, widget, wiring

**Files:**
- Modify: `frontend/ui.py`

No unit tests for the Gradio UI layer (Gradio event wiring is integration-tested manually). Verify via manual smoke test described in Step 6.

- [ ] **Step 1: Add the import for `build_qwen_prompt` and the new workflow constant**

At the top of `frontend/ui.py`, the existing imports include `from backend.core.tryon import run_tryon`. Change it to:

```python
from backend.core.tryon import build_qwen_prompt, run_tryon
```

After the existing `WORKFLOW_FASHN` line, add:

```python
WORKFLOW_QWEN = "phase_2/image_qwen_image_edit_2511_api.json"
```

- [ ] **Step 2: Update `on_upload` to output `catalogue_results`**

Replace the current `on_upload` function:

```python
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
```

Note: one extra return value (`results`) is appended at the end.

- [ ] **Step 3: Update `on_select` to auto-fill the Qwen prompt**

Replace the current `on_select` function:

```python
def on_select(user_photo, gallery_data: list, catalogue_results: list, evt: gr.SelectData):
    """Called when user clicks a recommendation. Shows customisation panel."""
    if user_photo is None or not gallery_data:
        return gr.update(visible=False), None, gr.update()

    selected_path, _ = gallery_data[evt.index]
    meta = catalogue_results[evt.index]["metadata"] if catalogue_results else {}
    qwen_prompt = build_qwen_prompt(meta)
    return gr.update(visible=True), Image.open(selected_path), gr.update(value=qwen_prompt)
```

- [ ] **Step 4: Update `switch_workflow` to cycle through 3 states**

Replace the current `switch_workflow` function:

```python
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
```

- [ ] **Step 5: Update `on_try_on` to handle Qwen**

Replace the current `on_try_on` function:

```python
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
```

- [ ] **Step 6: Update `build_ui` — add new state, widget, and rewire events**

Replace the `build_ui` function with:

```python
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
```

- [ ] **Step 7: Smoke test the UI**

```bash
uv run python main.py
```

Walk through this checklist manually:
1. Upload a photo → gallery appears
2. Click a gallery item → customisation panel appears, `prompt_input` is hidden (Flux mode)
3. Click "Switch to FasHN-VTO" → label updates, `category_input` appears, others hidden
4. Click "Switch to Qwen" → label updates, `prompt_input` appears with auto-filled text
5. Click "Switch to Flux2-Klein" → back to colour input
6. With Qwen active, select a gallery item → `prompt_input` auto-fills with garment annotation
7. (Optional, if ComfyUI running) click "Try it on" with each workflow

- [ ] **Step 8: Commit**

```bash
git add frontend/ui.py
git commit -m "feat: add Qwen workflow to UI — 3-state toggle, catalogue_results state, prompt auto-fill"
```

---

### Task 4: Final regression check

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS with no warnings about unexpected keyword arguments

- [ ] **Step 2: Commit if any minor fixes were needed; otherwise done**

```bash
git log --oneline -5
```

Verify the three feature commits are present:
- `feat: add Qwen-Image-Edit-2511 inject function and prompt builder`
- `feat: extend run_tryon with qwen workflow branch`
- `feat: add Qwen workflow to UI — 3-state toggle, catalogue_results state, prompt auto-fill`
