import json
import pytest  # noqa: F401 — used for pytest.raises
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
import io


def make_api_template():
    """Minimal tryon_api.json structure matching actual exported workflow."""
    return {
        "76": {"class_type": "LoadImage", "inputs": {"image": "old_user.webp"}},
        "132": {"class_type": "LoadImage", "inputs": {"image": "old_item.jpg"}},
        "92:113": {"class_type": "CLIPTextEncode", "inputs": {"text": "try on", "clip": ["92:111", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["92", 0], "filename_prefix": "Flux2-Klein-4b-base"}},
    }


def test_inject_params_modifies_correct_nodes(tmp_path):
    from tryon import inject_params

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
    from tryon import inject_params

    template = make_api_template()
    original_user_img = template["76"]["inputs"]["image"]

    inject_params(
        template,
        user_image_name="new_user.png",
        item_image_name="new_item.png",
        colour_prompt="in blue",
        user_node_id="76",
        item_node_id="132",
        prompt_node_id="92:113",
    )

    assert template["76"]["inputs"]["image"] == original_user_img


def test_upload_image_posts_multipart_and_returns_name():
    from tryon import upload_image

    fake_response = MagicMock()
    fake_response.json.return_value = {"name": "abc123.png", "subfolder": "", "type": "input"}
    fake_response.raise_for_status = MagicMock()

    img = Image.new("RGB", (64, 64), color=(0, 128, 255))

    with patch("tryon.requests.post", return_value=fake_response) as mock_post:
        name = upload_image(img, "http://127.0.0.1:8188")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://127.0.0.1:8188/upload/image"
    assert name == "abc123.png"


FAKE_IMG_INFO = {"filename": "Flux2-Klein-4b-base_00001_.png", "type": "output", "subfolder": ""}
FAKE_FASHN_INFO = {"filename": "ComfyUI_temp_abc.png", "type": "temp", "subfolder": ""}


def make_fashn_api_template():
    """Minimal FasHN-VTO_api.json structure for testing."""
    return {
        "2": {"class_type": "LoadImage", "inputs": {"image": "old_person.png"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "old_garment.jpg"}},
        "5": {"class_type": "FashnVtonInference", "inputs": {"category": "tops", "pipeline": ["4", 0], "person_image": ["2", 0], "garment_image": ["3", 0]}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    }


def test_run_tryon_flux_returns_pil_image(tmp_path):
    from tryon import run_tryon

    api_template_path = tmp_path / "tryon_api.json"
    with open(api_template_path, "w") as f:
        json.dump(make_api_template(), f)

    user_img = Image.new("RGB", (64, 64))
    item_img = Image.new("RGB", (64, 64))

    with patch("tryon.upload_image", side_effect=["user.png", "item.png"]) as mock_upload, \
         patch("tryon.submit_prompt", return_value="prompt-id-123") as mock_submit, \
         patch("tryon.poll_result", return_value=FAKE_IMG_INFO) as mock_poll, \
         patch("tryon.fetch_output_image", return_value=Image.new("RGB", (512, 512))) as mock_fetch:

        result = run_tryon(
            user_img=user_img,
            item_img=item_img,
            colour="in navy blue",
            workflow="flux",
            comfyui_url="http://127.0.0.1:8188",
            api_template_path=api_template_path,
        )

    assert isinstance(result, Image.Image)
    assert mock_upload.call_count == 2
    mock_submit.assert_called_once()
    mock_poll.assert_called_once_with("prompt-id-123", "http://127.0.0.1:8188")
    mock_fetch.assert_called_once_with(FAKE_IMG_INFO, "http://127.0.0.1:8188")


def test_run_tryon_fashn_injects_category(tmp_path):
    from tryon import run_tryon

    api_template_path = tmp_path / "fashn_api.json"
    with open(api_template_path, "w") as f:
        json.dump(make_fashn_api_template(), f)

    user_img = Image.new("RGB", (64, 64))
    item_img = Image.new("RGB", (64, 64))

    with patch("tryon.upload_image", side_effect=["user.png", "item.png"]), \
         patch("tryon.submit_prompt", return_value="pid-fashn") as mock_submit, \
         patch("tryon.poll_result", return_value=FAKE_FASHN_INFO), \
         patch("tryon.fetch_output_image", return_value=Image.new("RGB", (512, 512))):

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
    # Verify the submitted workflow had correct node values
    submitted_workflow = mock_submit.call_args[0][0]
    assert submitted_workflow["2"]["inputs"]["image"] == "user.png"
    assert submitted_workflow["3"]["inputs"]["image"] == "item.png"
    assert submitted_workflow["5"]["inputs"]["category"] == "bottoms"


def test_inject_params_fashn_validates_category():
    from tryon import inject_params_fashn

    template = make_fashn_api_template()
    with pytest.raises(ValueError, match="category must be one of"):
        inject_params_fashn(template, "u.png", "g.png", "invalid-category")


def test_inject_params_fashn_does_not_mutate_template():
    from tryon import inject_params_fashn

    template = make_fashn_api_template()
    original = template["2"]["inputs"]["image"]
    inject_params_fashn(template, "new_user.png", "new_garment.png", "tops")
    assert template["2"]["inputs"]["image"] == original
