import pytest
import requests
import numpy as np

from typing import Literal

INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
OPEN_CLIP_MODELS = [
    # ("ViT-B-32", "laion2b_s34b_b79k"),
    ("hf-hub:timm/ViT-B-16-SigLIP-i18n-256", None),
    ("ViT-B-32-SigLIP2-256", "webli"),
    # ("RN50", "openai"), # TODO: Fix RN50 models, error with ONNX into Triton
    ("convnext_base_w", "laion2b_s13b_b82k"),
]


def check_server_running():
    try:
        response = requests.get(f"{INFERENCE_BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_clip_model(model_name: str, pretrained: str):
    response = requests.post(
        f"{MODEL_BASE_URL}/load_clip_model",
        json={"name": model_name, "pretrained": pretrained},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to load model {model_name} with pretrained {pretrained}: {response.text}"
        )


def unload_clip_model(model_name: str, pretrained: str):
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model",
        json={"name": model_name, "pretrained": pretrained},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to unload model {model_name} with pretrained {pretrained}: {response.text}"
        )


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
            f"{MODEL_BASE_URL}/load_clip_model",
            json={"name": model_name, "pretrained": pretrained},
        )
        assert response.status_code == 200
        assert (
            "loaded successfully" in response.json()["message"]
            or "already loaded" in response.json()["message"]
        )
        unload_clip_model(model_name, pretrained)


@pytest.mark.integration
def test_infer_text_clip_batch():
    check_server_running()
    test_text = [
        "This is a test sentence.",
        "This is another test sentence.",
        "This is a third test sentence.",
    ]

    for model_name, pretrained in OPEN_CLIP_MODELS:
        load_clip_model(model_name, pretrained)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/infer_text",
            json={
                "name": model_name,
                "pretrained": pretrained,
                "text": test_text,
            },
        )
        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name, pretrained)


@pytest.mark.integration
def test_infer_image():
    check_server_running()
    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    for model_name, pretrained in OPEN_CLIP_MODELS:
        load_clip_model(model_name, pretrained)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/infer_image",
            json={
                "name": model_name,
                "pretrained": pretrained,
                "image": test_image,
            },
        )
        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name, pretrained)


@pytest.mark.integration
def test_infer_image_batch():
    check_server_running()
    test_image = [
        "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    ] * 3
    for model_name, pretrained in OPEN_CLIP_MODELS:
        load_clip_model(model_name, pretrained)
        response = requests.post(
            f"{INFERENCE_BASE_URL}/infer_image",
            json={
                "name": model_name,
                "pretrained": pretrained,
                "image": test_image,
            },
        )

        assert response.status_code == 200
        assert "embeddings" in response.json()
        assert "processingTimeMs" in response.json()

        unload_clip_model(model_name, pretrained)
