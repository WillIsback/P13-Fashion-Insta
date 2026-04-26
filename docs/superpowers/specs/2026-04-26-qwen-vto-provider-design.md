# Design: Qwen-Image-Edit-2511 Virtual Try-On Provider

**Date:** 2026-04-26
**Status:** Approved

## Overview

Add Qwen-Image-Edit-2511 as a third VTO provider alongside Flux2-Klein and FasHN-VTO. The model runs through ComfyUI, takes a person image, a garment image, and a text prompt. The prompt is auto-generated from catalogue garment annotations and is user-editable. The UI extends the existing 2-state workflow toggle to a 3-state cycle.

## Architecture

No new modules. Changes are confined to two existing files:
- `backend/core/tryon.py` — new inject function, prompt builder, `run_tryon` branch
- `frontend/ui.py` — new state, new UI element, extended toggle cycle

ComfyUI API workflow template: `phase_2/image_qwen_image_edit_2511_api.json` (already present).

## Backend (`backend/core/tryon.py`)

### Constants

```python
QWEN_PERSON_NODE  = "41"
QWEN_GARMENT_NODE = "83"
QWEN_PROMPT_NODE  = "170:151"
```

### `inject_params_qwen(template, user_image_name, item_image_name, prompt) -> dict`

Deep-copies the template and injects:
- `workflow[QWEN_PERSON_NODE]["inputs"]["image"]  = user_image_name`
- `workflow[QWEN_GARMENT_NODE]["inputs"]["image"] = item_image_name`
- `workflow[QWEN_PROMPT_NODE]["inputs"]["prompt"] = prompt`

Follows the same contract as `inject_params_flux` and `inject_params_fashn` (no mutation of the template).

### `build_qwen_prompt(meta: dict) -> str`

Builds a preservation-focused instruction from catalogue metadata:

```
"Virtual try-on: dress the person with the {garment} shown in the reference image.
Preserve the person's face, hair, skin tone, body pose, and the original background
exactly. Only replace the clothing."
```

Where `garment = meta.get("category_name", "garment").replace("_", " ").lower()`.

Rationale: the bare "Virtual Try On red dress" prompt caused the model to regenerate the background. Explicit preservation instructions fix this.

### `run_tryon` extension

New `prompt: str = ""` kwarg (ignored by flux and fashn branches).

New branch:
```python
elif workflow == "qwen":
    wf = inject_params_qwen(template, user_name, item_name, prompt or build_qwen_prompt({}))
```

`api_template_path` must point to `phase_2/image_qwen_image_edit_2511_api.json` when `workflow="qwen"`.

## Frontend (`frontend/ui.py`)

### New constant

```python
WORKFLOW_QWEN = "phase_2/image_qwen_image_edit_2511_api.json"
```

### New state: `catalogue_results` (`gr.State`)

Stores the raw `list[dict]` returned by `recommend()` (each dict has `score` and `metadata` keys). Set by `on_upload`, consumed by `on_select`.

### `on_upload` changes

Returns one additional output: `catalogue_results` (the raw results list).

### `on_select` changes

- New input: `catalogue_results`
- Looks up `catalogue_results[evt.index]["metadata"]`
- Returns one additional output: `gr.update(value=build_qwen_prompt(meta))` targeting `prompt_input`

Import: `from backend.core.tryon import build_qwen_prompt` added to frontend imports.

### New UI element: `prompt_input`

```python
prompt_input = gr.Textbox(
    label="Try-on prompt (Qwen)",
    placeholder="Auto-filled from garment annotations — edit freely",
    visible=False,
)
```

Placed in the customisation column alongside `colour_input` and `category_input`.

### `switch_workflow` — 3-state cycle

| Current state | Next state | Button label | Visible input |
|---------------|-----------|--------------|---------------|
| Flux2-Klein | FasHN-VTO | "Switch to Qwen" | `category_input` |
| FasHN-VTO | Qwen | "Switch to Flux2-Klein" | `prompt_input` |
| Qwen | Flux2-Klein | "Switch to FasHN-VTO" | `colour_input` |

Returns visibility updates for all three input controls.

### `on_try_on` changes

- New input: `prompt_input`
- Passes `prompt=prompt_input.strip()` to `run_tryon` when `workflow_state["name"] == "Qwen"`
- `workflow="qwen"` and `api_template_path=Path(WORKFLOW_QWEN)` for Qwen branch

### Unchanged

Gallery, size dropdown, approve/reject buttons, error box, `_check_index`, `find_image_path`.

## Data Flow

```
on_upload → recommend() → catalogue_results (State) + gallery_data (State)
on_select(catalogue_results, evt) → selected_preview + prompt_input (auto-filled)
switch_workflow(workflow_state) → workflow_state + toggle label + 3x visibility
on_try_on(workflow_state, prompt_input, ...) → run_tryon(workflow="qwen", prompt=...)
                                             → inject_params_qwen → ComfyUI
```

## Error Handling

No new error cases. Existing `ConnectionError`, `TimeoutError`, and generic `Exception` catches in `on_try_on` cover all three workflows uniformly.

## Testing

- `test_inject_params_qwen_modifies_correct_nodes` — verifies person, garment, prompt injection
- `test_inject_params_qwen_does_not_mutate_template` — immutability check
- `test_build_qwen_prompt_uses_category_name` — verifies garment name appears in output
- `test_run_tryon_qwen_returns_pil_image` — end-to-end mock (mirrors existing flux/fashn tests)
