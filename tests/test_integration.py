import pytest
import requests
import numpy as np

from typing import Literal

INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
SENTENCE_TRANSFORMER_MODEL_EMBEDDING_SIZE = 384
OPENCLIP_MODEL = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CUSTOM_TEXT_CLIP_MODEL = "hf-hub:timm/ViT-B-32-SigLIP2-256"


def check_server_running():
    try:
        response = requests.get(f"{INFERENCE_BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_openclip_model(clip_type: Literal["CLIP", "CustomTextCLIP"] = "CLIP"):
    if clip_type == "CLIP":
        model_name = OPENCLIP_MODEL
    elif clip_type == "CustomTextCLIP":
        model_name = CUSTOM_TEXT_CLIP_MODEL
    else:
        raise ValueError("Invalid model type. Must be 'CLIP' or 'CustomTextCLIP'.")

    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": model_name, "library": "open_clip"},
    )
    response.raise_for_status()


def load_sentence_transformer_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "library": "sentence_transformers"},
    )
    response.raise_for_status()


def unload_sentence_transformer_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL},
    )
    response.raise_for_status()


def unload_openclip_model(clip_type: Literal["CLIP", "CustomTextCLIP"] = "CLIP"):
    if clip_type == "CLIP":
        model_name = OPENCLIP_MODEL
    elif clip_type == "CustomTextCLIP":
        model_name = CUSTOM_TEXT_CLIP_MODEL
    else:
        raise ValueError("Invalid model type. Must be 'CLIP' or 'CustomTextCLIP'.")

    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model",
        json={"name": model_name, "library": "open_clip"},
    )
    response.raise_for_status()


@pytest.mark.integration
def test_health():
    check_server_running()
    response = requests.get(f"{INFERENCE_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "The inference server is running."}


@pytest.mark.integration
def test_load_sentence_transformer_model():
    check_server_running()
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "library": "sentence_transformers"},
    )
    assert response.status_code == 200
    assert (
        "loaded successfully" in response.json()["message"]
        or "already loaded" in response.json()["message"]
    )
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_load_loaded_sentence_transformer_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "library": "sentence_transformers"},
    )
    assert response.status_code == 200
    assert "already loaded" in response.json()["message"]
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_load_clip_model():
    check_server_running()
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": OPENCLIP_MODEL, "library": "open_clip"},
    )
    assert response.status_code == 200
    assert (
        "loaded successfully" in response.json()["message"]
        or "already loaded" in response.json()["message"]
    )
    unload_openclip_model()


@pytest.mark.integration
def test_load_custom_text_clip_model():
    check_server_running()
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={
            "name": CUSTOM_TEXT_CLIP_MODEL,
            "library": "open_clip",
        },
    )
    assert response.status_code == 200
    assert (
        "loaded successfully" in response.json()["message"]
        or "already loaded" in response.json()["message"]
    )
    unload_openclip_model(clip_type="CustomTextCLIP")


@pytest.mark.integration
def test_infer_text():
    check_server_running()
    load_sentence_transformer_model()
    test_text = "This is a test sentence."
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "text": test_text},
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_infer_text_truncated():
    check_server_running()
    load_sentence_transformer_model()
    test_text = "This is a test sentence."
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "text": test_text, "nDims": 128},
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()
    assert len(response.json()["embeddings"][0]) == 128
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_infer_text_batch():
    check_server_running()
    load_sentence_transformer_model()
    test_text = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "text": test_text},
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_infer_text_clip_batch():
    check_server_running()
    load_openclip_model()
    test_text = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={
            "name": OPENCLIP_MODEL,
            "text": test_text,
        },
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()
    unload_openclip_model()


@pytest.mark.integration
def test_infer_image():
    check_server_running()
    load_openclip_model()
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_image",
        json={
            "name": OPENCLIP_MODEL,
            "image": test_image,
        },
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()
    unload_openclip_model()


@pytest.mark.integration
def test_infer_image_batch():
    check_server_running()
    load_openclip_model()
    test_image = [
        "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    ] * 3
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_image",
        json={
            "name": OPENCLIP_MODEL,
            "image": test_image,
        },
    )

    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()


@pytest.mark.integration
def test_infer_text_image():
    check_server_running()
    load_openclip_model()

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]

    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed",
        json={
            "name": OPENCLIP_MODEL,
            "text": test_texts,
            "image": test_image,
        },
    )

    assert response.status_code == 200
    assert "textEmbeddings" in response.json()
    assert "imageEmbeddings" in response.json()
    assert len(response.json()["textEmbeddings"]) == len(test_texts)

    image_embeddings_arr = np.array(response.json()["imageEmbeddings"])
    text_embeddings_arr = np.array(response.json()["textEmbeddings"])

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]

    unload_openclip_model()


@pytest.mark.integration
def test_unload_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model", json={"name": SENTENCE_TRANSFORMER_MODEL}
    )
    assert response.status_code == 200
    assert "unloaded successfully" in response.json()["message"]


@pytest.mark.integration
def test_unload_and_load_sentence_transformer_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model", json={"name": SENTENCE_TRANSFORMER_MODEL}
    )
    assert response.status_code == 200
    assert "unloaded successfully" in response.json()["message"]
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "library": "sentence_transformers"},
    )
    assert response.status_code == 200
    assert "loaded successfully" in response.json()["message"]
    unload_sentence_transformer_model()


@pytest.mark.integration
def test_unload_and_load_clip_model():
    check_server_running()
    load_openclip_model()
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model", json={"name": OPENCLIP_MODEL}
    )
    assert response.status_code == 200
    assert "unloaded successfully" in response.json()["message"]
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": OPENCLIP_MODEL, "library": "open_clip"},
    )
    assert response.status_code == 200
    assert "loaded successfully" in response.json()["message"]

    unload_openclip_model()


@pytest.mark.integration
def test_delete_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.delete(
        f"{MODEL_BASE_URL}/delete_model", json={"name": SENTENCE_TRANSFORMER_MODEL}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


@pytest.mark.integration
def test_delete_clip_model():
    check_server_running()
    load_openclip_model()
    response = requests.delete(
        f"{MODEL_BASE_URL}/delete_model",
        json={"name": OPENCLIP_MODEL},
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


@pytest.mark.integration
def test_loaded_models():
    check_server_running()
    load_sentence_transformer_model()
    load_openclip_model()
    response = requests.get(f"{MODEL_BASE_URL}/loaded_models")
    assert response.status_code == 200
    assert "models" in response.json()

    unload_sentence_transformer_model()
    unload_openclip_model()


@pytest.mark.integration
def test_repository_models():
    check_server_running()
    response = requests.get(f"{MODEL_BASE_URL}/repository_models")
    assert response.status_code == 200
    assert "models" in response.json()


@pytest.mark.integration
def test_metrics():
    check_server_running()
    response = requests.get(f"{INFERENCE_BASE_URL}/metrics")
    assert response.status_code == 200
    assert "modelStats" in response.json()


@pytest.mark.integration
def test_embedding_size_endpoint():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.get(
        f"{MODEL_BASE_URL}/model_embedding_size",
        params={"name": SENTENCE_TRANSFORMER_MODEL},
    )
    assert response.status_code == 200
    assert "embeddingSize" in response.json()
    assert response.json()["embeddingSize"] == SENTENCE_TRANSFORMER_MODEL_EMBEDDING_SIZE

    unload_sentence_transformer_model()
