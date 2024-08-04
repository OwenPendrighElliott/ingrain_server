import pytest
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8686"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
OPENCLIP_MODEL = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"


def check_server_running():
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_openclip_model():
    response = requests.post(
        f"{BASE_URL}/load_clip_model",
        json={"model_name": OPENCLIP_MODEL, "pretrained": OPENCLIP_PRETRAINED},
    )
    response.raise_for_status()


def load_sentence_transformer_model():
    response = requests.post(
        f"{BASE_URL}/load_sentence_transformer_model",
        json={"model_name": SENTENCE_TRANSFORMER_MODEL},
    )
    response.raise_for_status()


@pytest.mark.integration
def test_health():
    check_server_running()
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "The server is running."}


@pytest.mark.integration
def test_load_sentence_transformer_model():
    check_server_running()
    response = requests.post(
        f"{BASE_URL}/load_sentence_transformer_model",
        json={"model_name": SENTENCE_TRANSFORMER_MODEL},
    )
    assert response.status_code == 200
    assert (
        "loaded successfully" in response.json()["message"]
        or "already loaded" in response.json()["message"]
    )


@pytest.mark.integration
def test_load_loaded_sentence_transformer_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{BASE_URL}/load_sentence_transformer_model",
        json={"model_name": SENTENCE_TRANSFORMER_MODEL},
    )
    assert response.status_code == 200
    assert "already loaded" in response.json()["message"]


@pytest.mark.integration
def test_load_clip_model():
    check_server_running()
    response = requests.post(
        f"{BASE_URL}/load_clip_model",
        json={"model_name": OPENCLIP_MODEL, "pretrained": OPENCLIP_PRETRAINED},
    )
    assert response.status_code == 200
    assert "loaded successfully" in response.json()["message"]


@pytest.mark.integration
def test_infer_text():
    check_server_running()
    test_text = "This is a test sentence."
    response = requests.post(
        f"{BASE_URL}/infer_text",
        json={"model_name": SENTENCE_TRANSFORMER_MODEL, "text": test_text},
    )
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert "processingTimeMs" in response.json()


@pytest.mark.integration
def test_infer_image():
    check_server_running()
    load_openclip_model()
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = requests.post(
        f"{BASE_URL}/infer_image",
        json={
            "model_name": OPENCLIP_MODEL,
            "pretrained": OPENCLIP_PRETRAINED,
            "image": test_image,
        },
    )
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert "processingTimeMs" in response.json()


@pytest.mark.integration
def test_infer_text_image():
    check_server_running()
    load_openclip_model()

    # this image is pink
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACcklEQVR4nOzSMRHAIADAwF4PbVjFITMOWMnwryBDxp7rg6r/dQDcGJQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaQYlzaCkGZQ0g5JmUNIMSppBSTMoaScAAP//3nYDppOW6x0AAAAASUVORK5CYII="
    test_texts = ["A green image", "A pink image"]

    response = requests.post(
        f"{BASE_URL}/infer",
        json={
            "model_name": OPENCLIP_MODEL,
            "pretrained": OPENCLIP_PRETRAINED,
            "text": test_texts,
            "image": test_image,
        },
    )

    assert response.status_code == 200
    assert "text_embeddings" in response.json()
    assert "image_embeddings" in response.json()
    assert len(response.json()["text_embeddings"]) == len(test_texts)

    image_embeddings_arr = np.array(response.json()["image_embeddings"])
    text_embeddings_arr = np.array(response.json()["text_embeddings"])

    image_text_similarities = np.dot(image_embeddings_arr, text_embeddings_arr.T)
    assert image_text_similarities[0, 0] < image_text_similarities[0, 1]


@pytest.mark.integration
def test_unload_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{BASE_URL}/unload_model", json={"model_name": SENTENCE_TRANSFORMER_MODEL}
    )
    assert response.status_code == 200
    assert "unloaded successfully" in response.json()["message"]


@pytest.mark.integration
def test_delete_model():
    check_server_running()
    load_sentence_transformer_model()
    response = requests.post(
        f"{BASE_URL}/delete_model", json={"model_name": SENTENCE_TRANSFORMER_MODEL}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


@pytest.mark.integration
def test_delete_clip_model():
    check_server_running()
    load_openclip_model()
    response = requests.post(
        f"{BASE_URL}/delete_model",
        json={"model_name": OPENCLIP_MODEL, "pretrained": OPENCLIP_PRETRAINED},
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


@pytest.mark.integration
def test_loaded_models():
    check_server_running()
    load_sentence_transformer_model()
    load_openclip_model()
    response = requests.get(f"{BASE_URL}/loaded_models")
    assert response.status_code == 200
    assert "models" in response.json()


@pytest.mark.integration
def test_repository_models():
    check_server_running()
    response = requests.get(f"{BASE_URL}/repository_models")
    assert response.status_code == 200
    assert "models" in response.json()


@pytest.mark.integration
def test_metrics():
    check_server_running()
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "model_stats" in response.json()