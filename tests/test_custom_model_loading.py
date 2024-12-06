import pytest
import requests

# to run these tests you must create a file server to serve the custom model files
FILE_SERVER_URL = "http://localhost:9999"
MODEL_SERVER_URL = "http://localhost:8687"
INFERENCE_SERVER_URL = "http://localhost:8686"


def load_custom_model_helper(
    library: str, model_name: str, pretrained: str
) -> requests.Response:
    if library == "open_clip":
        url = f"{MODEL_SERVER_URL}/load_clip_model"
    elif library == "timm":
        url = f"{MODEL_SERVER_URL}/load_timm_model"
    elif library == "sentence_transformers":
        url = f"{MODEL_SERVER_URL}/load_sentence_transformer_model"
    else:
        raise ValueError("Invalid library name")

    body = {
        "name": model_name,
        "pretrained": pretrained,
    }

    response = requests.post(url, json=body)
    return response


def download_timm_model():
    timm_model_url = f"{FILE_SERVER_URL}/tinypatch16timm/model.safetensors"

    body = {
        "library": "timm",
        "pretrained_name": "custom_tiny_patch16",
        "safetensors_url": timm_model_url,
        "num_classes": 21843,
    }

    response = requests.post(f"{MODEL_SERVER_URL}/download_custom_model", json=body)
    return response


def download_sentence_transformers_model():
    sentence_transformers_model_url = f"{FILE_SERVER_URL}/e5smallv2/model.safetensors"
    config_json_url = f"{FILE_SERVER_URL}/e5smallv2/config.json"
    tokenizer_json_url = f"{FILE_SERVER_URL}/e5smallv2/tokenizer.json"
    tokenizer_config_json_url = f"{FILE_SERVER_URL}/e5smallv2/tokenizer_config.json"
    vocab_txt_url = f"{FILE_SERVER_URL}/e5smallv2/vocab.txt"
    special_tokens_map_json_url = f"{FILE_SERVER_URL}/e5smallv2/special_tokens_map.json"
    pooling_config_json_url = f"{FILE_SERVER_URL}/e5smallv2/1_Pooling_config.json"
    sentence_bert_config_json_url = (
        f"{FILE_SERVER_URL}/e5smallv2/sentence_bert_config.json"
    )
    modules_json_url = f"{FILE_SERVER_URL}/e5smallv2/modules.json"

    body = {
        "library": "sentence_transformers",
        "pretrained_name": "custom-e5-small-v2",
        "safetensors_url": sentence_transformers_model_url,
        "config_json_url": config_json_url,
        "tokenizer_json_url": tokenizer_json_url,
        "tokenizer_config_json_url": tokenizer_config_json_url,
        "vocab_txt_url": vocab_txt_url,
        "special_tokens_map_json_url": special_tokens_map_json_url,
        "pooling_config_json_url": pooling_config_json_url,
        "sentence_bert_config_json_url": sentence_bert_config_json_url,
        "modules_json_url": modules_json_url,
    }

    response = requests.post(f"{MODEL_SERVER_URL}/download_custom_model", json=body)
    return response


def download_open_clip_model():
    open_clip_model_url = f"{FILE_SERVER_URL}/mobile_clip/model.safetensors"

    body = {
        "library": "open_clip",
        "pretrained_name": "custom_mobile_clip",
        "safetensors_url": open_clip_model_url,
    }

    response = requests.post(f"{MODEL_SERVER_URL}/download_custom_model", json=body)
    return response


@pytest.mark.custom_model
def test_download_open_clip_model():
    response = download_open_clip_model()
    print(response.status_code)
    print(response.text)
    print(response.json())
    assert response.status_code == 200
    assert "custom_mobile_clip" in response.json()["message"]
    assert "downloaded successfully" in response.json()["message"]


@pytest.mark.custom_model
def test_download_timm_model():
    response = download_timm_model()
    assert response.status_code == 200
    assert "custom_tiny_patch16" in response.json()["message"]
    assert "downloaded successfully" in response.json()["message"]


@pytest.mark.custom_model
def test_download_sentence_transformers_model():
    response = download_sentence_transformers_model()
    assert response.status_code == 200
    assert "custom-e5-small-v2" in response.json()["message"]
    assert "downloaded successfully" in response.json()["message"]


@pytest.mark.custom_model
def test_custom_sentence_transformer_inference():
    download_sentence_transformers_model()

    response = load_custom_model_helper(
        "sentence_transformers", "custom-e5-small-v2", None
    )

    assert response.status_code == 200

    body = {
        "name": "custom-e5-small-v2",
        "text": "Hello, how are you?",
    }

    response = requests.post(f"{INFERENCE_SERVER_URL}/infer_text", json=body)
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert len(response.json()["embeddings"][0]) == 384


@pytest.mark.custom_model
def test_custom_open_clip_inference():
    download_open_clip_model()

    response = load_custom_model_helper(
        "open_clip", "MobileCLIP-S1", "custom_mobile_clip"
    )

    assert response.status_code == 200

    body = {
        "name": "MobileCLIP-S1",
        "pretrained": "custom_mobile_clip",
        "text": "Hello, how are you?",
    }

    response = requests.post(f"{INFERENCE_SERVER_URL}/infer_text", json=body)
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert len(response.json()["embeddings"][0]) == 512


@pytest.mark.custom_model
def test_custom_timm_inference():
    download_timm_model()

    response = load_custom_model_helper(
        "timm", "vit_tiny_patch16_224", "custom_tiny_patch16"
    )

    assert response.status_code == 200
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="

    body = {
        "name": "vit_tiny_patch16_224",
        "pretrained": "custom_tiny_patch16",
        "image": test_image,
    }

    response = requests.post(f"{INFERENCE_SERVER_URL}/infer_image", json=body)

    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert len(response.json()["embeddings"][0]) == 21843
