from ingrain_inference_server.ingrain_common.common import (
    get_model_name,
    get_text_image_model_names,
)


def test_get_model_name():
    assert get_model_name("model/name") == "model_name"
    assert get_model_name("model/name", "pretrained") == "model_name_pretrained"
    assert get_model_name("model/name", None) == "model_name"


def test_hf_hub_model_name():
    assert (
        get_model_name("hf-hub:timm/ViT-b-16-SigLIP2-256", None)
        == "hf-hub___timm_ViT-b-16-SigLIP2-256"
    )


def test_get_text_image_model_names():
    text_encoder, image_encoder = get_text_image_model_names("model/name")
    assert text_encoder == "model_name_text_encoder"
    assert image_encoder == "model_name_image_encoder"

    text_encoder, image_encoder = get_text_image_model_names("model/name", "pretrained")
    assert text_encoder == "model_name_pretrained_text_encoder"
    assert image_encoder == "model_name_pretrained_image_encoder"
