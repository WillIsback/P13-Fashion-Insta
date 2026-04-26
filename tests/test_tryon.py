import json
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock


def make_api_template():
    return {
        "76": {"class_type": "LoadImage", "inputs": {"image": "old_user.webp"}},
        "132": {"class_type": "LoadImage", "inputs": {"image": "old_item.jpg"}},
        "92:113": {"class_type": "CLIPTextEncode", "inputs": {"text": "try on", "clip": ["92:111", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["92", 0], "filename_prefix": "Flux2-Klein-4b-base"}},
    }


def make_fashn_api_template():
    return {
        "2": {"class_type": "LoadImage", "inputs": {"image": "old_person.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "old_garment.jpg"}},
        "5": {"class_type": "FashnVtonInference", "inputs": {"category": "tops", "pipeline": ["4", 0], "person_image": ["2", 0], "garment_image": ["3", 0]}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    }


FAKE_IMG_INFO = {"filename": "Flux2-Klein-4b-base_00001_.png", "type": "output", "subfolder": ""}
FAKE_FASHN_INFO = {"filename": "ComfyUI_temp_abc.png", "type": "temp", "subfolder": ""}


def test_inject_params_modifies_correct_nodes():
    from backend.core.tryon import inject_params

    template = make_api_template()
    result = inject_params(
        template,
        user_image_name="user123.png",
        item_image_name="item456.png",
        colour_prompt="in red",
        user_node_id="76",
        item_node_id="132",
        prompt_node_id="92:113",
    )

    assert result["76"]["inputs"]["image"] == "user123.png"
    assert result["132"]["inputs"]["image"] == "item456.png"
    assert "in red" in result["92:113"]["inputs"]["text"]


def test_inject_params_does_not_mutate_template():
    from backend.core.tryon import inject_params

    template = make_api_template()
    original = template["76"]["inputs"]["image"]
    inject_params(template, "new_user.png", "new_item.png", "in blue",
                  user_node_id="76", item_node_id="132", prompt_node_id="92:113")
    assert template["76"]["inputs"]["image"] == original


def test_upload_image_posts_multipart_and_returns_name():
    from backend.core.tryon import upload_image

    fake_response = MagicMock()
    fake_response.json.return_value = {"name": "abc123.png", "subfolder": "", "type": "input"}
    fake_response.raise_for_status = MagicMock()

    img = Image.new("RGB", (64, 64), color=(0, 128, 255))

    with patch("backend.core.tryon.requests.post", return_value=fake_response) as mock_post:
        name = upload_image(img, "http://127.0.0.1:8188")

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "http://127.0.0.1:8188/upload/image"
    assert name == "abc123.png"


def test_run_tryon_flux_returns_pil_image(tmp_path):
    from backend.core.tryon import run_tryon

    api_template_path = tmp_path / "tryon_api.json"
    api_template_path.write_text(json.dumps(make_api_template()))

    user_img = Image.new("RGB", (64, 64))
    item_img = Image.new("RGB", (64, 64))

    with patch("backend.core.tryon.upload_image", side_effect=["user.png", "item.png"]), \
         patch("backend.core.tryon.submit_prompt", return_value="prompt-id-123") as mock_submit, \
         patch("backend.core.tryon.poll_result", return_value=FAKE_IMG_INFO) as mock_poll, \
         patch("backend.core.tryon.fetch_output_image", return_value=Image.new("RGB", (512, 512))):

        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            colour="in navy blue",
            workflow="flux",
            comfyui_url="http://127.0.0.1:8188",
            api_template_path=api_template_path,
        )

    assert isinstance(result, Image.Image)
    mock_submit.assert_called_once()
    mock_poll.assert_called_once_with("prompt-id-123", "http://127.0.0.1:8188")


def test_run_tryon_fashn_injects_category(tmp_path):
    from backend.core.tryon import run_tryon

    api_template_path = tmp_path / "fashn_api.json"
    api_template_path.write_text(json.dumps(make_fashn_api_template()))

    user_img = Image.new("RGB", (64, 64))
    item_img = Image.new("RGB", (64, 64))

    with patch("backend.core.tryon.upload_image", side_effect=["user.png", "item.png"]), \
         patch("backend.core.tryon.submit_prompt", return_value="pid-fashn") as mock_submit, \
         patch("backend.core.tryon.poll_result", return_value=FAKE_FASHN_INFO), \
         patch("backend.core.tryon.fetch_output_image", return_value=Image.new("RGB", (512, 512))):

        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            colour="",
            category="bottoms",
            workflow="fashn",
            comfyui_url="http://127.0.0.1:8188",
            api_template_path=api_template_path,
        )

    assert isinstance(result, Image.Image)
    submitted_workflow = mock_submit.call_args[0][0]
    assert submitted_workflow["2"]["inputs"]["image"] == "user.png"
    assert submitted_workflow["3"]["inputs"]["image"] == "item.png"
    assert submitted_workflow["5"]["inputs"]["category"] == "bottoms"


def test_inject_params_fashn_validates_category():
    from backend.core.tryon import inject_params_fashn

    with pytest.raises(ValueError, match="category must be one of"):
        inject_params_fashn(make_fashn_api_template(), "u.png", "g.png", "invalid-category")


def test_inject_params_fashn_does_not_mutate_template():
    from backend.core.tryon import inject_params_fashn

    template = make_fashn_api_template()
    original = template["2"]["inputs"]["image"]
    inject_params_fashn(template, "new_user.png", "new_garment.png", "tops")
    assert template["2"]["inputs"]["image"] == original


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


def test_build_qwen_prompt_fallback_on_empty_category():
    from backend.core.tryon import build_qwen_prompt

    prompt = build_qwen_prompt({"category_name": ""})
    assert "garment" in prompt.lower()
    assert "  " not in prompt  # no double space


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
