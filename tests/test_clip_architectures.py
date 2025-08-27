import pytest
import requests

INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
OPEN_CLIP_MODELS = [
    ("hf-hub:timm/ViT-B-16-SigLIP-i18n-256", "open_clip"),
    ("hf-hub:timm/ViT-B-32-SigLIP2-256", "open_clip"),
    ("hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg", "open_clip"),
]

MODEL_EMBEDDING_SIZES = {
    "hf-hub:timm/ViT-B-16-SigLIP-i18n-256": 768,
    "hf-hub:timm/ViT-B-32-SigLIP2-256": 768,
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg": 640,
}


def check_server_running():
    try:
        response = requests.get(f"{INFERENCE_BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_clip_model(model_name: str, library: str):
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": model_name, "library": library},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Failed to load model {model_name}: {response.text}")


def unload_clip_model(model_name: str):
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model",
        json={"name": model_name},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Failed to unload model {model_name}: {response.text}")


@pytest.mark.integration
def test_health():
    check_server_running()
    response = requests.get(f"{INFERENCE_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "The inference server is running."}


@pytest.mark.integration
def test_load_clip_models():
    check_server_running()
    for model_name, pretrained in OPEN_CLIP_MODELS:
        response = requests.post(
            f"{MODEL_BASE_URL}/load_model",
            json={"name": model_name, "library": pretrained},
        )
        assert response.status_code == 200
        assert (
            "loaded successfully" in response.json()["message"]
            or "already loaded" in response.json()["message"]
        )
        unload_clip_model(model_name)


@pytest.mark.integration
def test_infer_text_clip_batch():
    check_server_running()
    test_text = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]

    for model_name, library in OPEN_CLIP_MODELS:
        load_clip_model(model_name, library)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/embed_text",
            json={
                "name": model_name,
                "text": test_text,
            },
        )
        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name)


@pytest.mark.integration
def test_infer_image():
    check_server_running()
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    for model_name, library in OPEN_CLIP_MODELS:
        load_clip_model(model_name, library)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/embed_image",
            json={
                "name": model_name,
                "image": test_image,
            },
        )

        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name)


@pytest.mark.integration
def test_infer_image_batch():
    check_server_running()
    test_image = [
        "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    ] * 3
    for model_name, library in OPEN_CLIP_MODELS:
        load_clip_model(model_name, library)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/embed_image",
            json={
                "name": model_name,
                "image": test_image,
            },
        )

        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name)


@pytest.mark.integration
def test_embedding_size_endpoint_clip():
    check_server_running()
    for model_name, library in OPEN_CLIP_MODELS:
        load_clip_model(model_name, library)
        response = requests.post(
            f"{MODEL_BASE_URL}/model_embedding_size",
            json={"name": model_name},
        )
        assert response.status_code == 200
        assert "embeddingSize" in response.json()
        target_size = MODEL_EMBEDDING_SIZES.get(model_name)
        assert response.json()["embeddingSize"] == target_size
        unload_clip_model(model_name)
