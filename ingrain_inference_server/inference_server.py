from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
import time
import asyncio
from ingrain_inference.api_models.request_models import (
    EmbeddingRequest,
    TextEmbeddingRequest,
    ImageEmbeddingRequest,
    ImageClassificationRequest,
)
from ingrain_inference.api_models.response_models import (
    EmbeddingResponse,
    TextEmbeddingResponse,
    ImageEmbeddingResponse,
    GenericMessageResponse,
    MetricsResponse,
    ImageClassificationResponse,
)
from ingrain_inference.inference.triton_open_clip.clip_inference import (
    TritonCLIPInferenceClient,
)
from ingrain_inference.inference.triton_sentence_transformers.sentence_transformer_inference import (
    TritonSentenceTransformersInferenceClient,
)
from ingrain_inference.inference.triton_timm.timm_inference import (
    TritonTimmInferenceClient,
)
from ingrain_common.common import get_model_name
from ingrain_inference.inference.model_cache import LRUModelCache
from threading import Lock
import tritonclient.grpc as grpcclient
import os
from typing import Union


TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
TRITON_MODEL_REPOSITORY_PATH = "model_repository"
CUSTOM_MODEL_DIR = "custom_model_files"

app = FastAPI(default_response_class=ORJSONResponse)

# Model cache and lock
MODEL_CACHE = LRUModelCache(capacity=5)
MODEL_CACHE_LOCK = Lock()

EMBEDDING_MODEL_LIBRARIES = ["open_clip", "sentence_transformers"]
CLASSIFICATION_MODEL_LIBRARIES = ["timm"]


def get_model_library(model_name: str, pretrained: Union[str, None]) -> str:
    friendly_name = get_model_name(model_name, pretrained)
    try:
        with open(
            os.path.join(
                TRITON_MODEL_REPOSITORY_PATH, friendly_name, "library_name.txt"
            ),
            "r",
        ) as f:
            return f.read().strip()
    except FileNotFoundError:
        pass

    try:
        with open(
            os.path.join(
                TRITON_MODEL_REPOSITORY_PATH,
                friendly_name + "_text_encoder",
                "library_name.txt",
            ),
            "r",
        ) as f:
            return f.read().strip()
    except FileNotFoundError:
        pass

    return None


def client_from_cache(model_name: str, pretrained: Union[str, None]) -> Union[
    TritonCLIPInferenceClient,
    TritonSentenceTransformersInferenceClient,
    TritonTimmInferenceClient,
    None,
]:
    cache_key = (model_name, pretrained)

    with MODEL_CACHE_LOCK:
        client = MODEL_CACHE.get(cache_key)

    # if this worker is aware of the model, check if it's ready
    if client is not None:
        if not client.is_ready():
            # if its not ready, remove it from the cache as this worker is out of sync
            with MODEL_CACHE_LOCK:
                MODEL_CACHE.remove(cache_key)
            return None

        return client

    nice_model_name = get_model_name(model_name, pretrained)
    model_library = get_model_library(model_name, pretrained)

    if not model_library:
        return None

    # if the model isn't in this workers cache, check if it's ready
    if (
        TRITON_CLIENT.is_model_ready(nice_model_name)
        and model_library == "sentence_transformers"
    ):
        # if the model is ready, create a client for it
        # the model name is used directly for sentence transformers
        client = TritonSentenceTransformersInferenceClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            custom_model_dir=CUSTOM_MODEL_DIR,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    if TRITON_CLIENT.is_model_ready(nice_model_name) and model_library == "timm":
        # if the model is ready, create a client for it
        # the model name is used directly for timm models
        client = TritonTimmInferenceClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    if (
        TRITON_CLIENT.is_model_ready(nice_model_name + "_text_encoder")
        and TRITON_CLIENT.is_model_ready(nice_model_name + "_image_encoder")
        and model_library == "open_clip"
    ):
        # if the model is ready, create a client for it
        # the model name must be split into text and image encoders for CLIP
        client = TritonCLIPInferenceClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            custom_model_dir=CUSTOM_MODEL_DIR,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    # if the model isn't ready, and the worker doesn't have it in cache, return None
    return None


@app.get("/health", response_model=GenericMessageResponse)
async def health() -> GenericMessageResponse:
    return GenericMessageResponse(message="The inference server is running.")


@app.post("/embed_text", response_model=TextEmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest) -> TextEmbeddingResponse:
    model_name = request.name
    text = request.text
    normalize = request.normalize
    n_dims = request.n_dims

    client = client_from_cache(model_name, None)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not loaded. Load the model first using /load_model on the model server.",
        )

    if "text" not in client.modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} does not support text inference.",
        )

    if client.library_name not in EMBEDDING_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not an embedding model.",
        )

    start = time.perf_counter()
    embedding = client.encode_text(text, normalize=normalize, n_dims=n_dims)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]

    return TextEmbeddingResponse(
        embeddings=embedding_list, processing_time_ms=(end - start) * 1000
    )


@app.post("/embed_image", response_model=ImageEmbeddingResponse)
async def embed_image(request: ImageEmbeddingRequest) -> ImageEmbeddingResponse:
    model_name = request.name
    images = request.image
    normalize = request.normalize
    n_dims = request.n_dims
    image_download_headers = request.image_download_headers

    client = client_from_cache(model_name, None)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not loaded. Load the model first using /load_model on the model server.",
        )

    if isinstance(images, str):
        images = [images]

    image_data = client.load_images_parallel(
        images, image_download_headers=image_download_headers
    )

    if image_data is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please provide a valid image URL or a base64 encoded image.",
        )

    if "image" not in client.modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} does not support image inference.",
        )

    if client.library_name not in EMBEDDING_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not an embedding model.",
        )

    start = time.perf_counter()
    embedding = client.encode_image(image_data, normalize=normalize, n_dims=n_dims)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]
    return ImageEmbeddingResponse(
        embeddings=embedding_list,
        processing_time_ms=(end - start) * 1000,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    model_name = request.name
    texts = request.text
    images = request.image
    normalize = request.normalize
    n_dims = request.n_dims
    image_download_headers = request.image_download_headers

    client = client_from_cache(model_name, None)

    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not loaded. Load the model first using /load_model on the model server.",
        )

    response = {}
    tasks = []

    if texts is not None:
        if "text" not in client.modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not support text inference.",
            )
        if isinstance(texts, str):
            texts = [texts]
        tasks.append(asyncio.to_thread(client.encode_text, texts, normalize, n_dims))

    if images is not None:
        if "image" not in client.modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not support image inference.",
            )
        if isinstance(images, str):
            images = [images]
        image_datas = client.load_images_parallel(
            images, image_download_headers=image_download_headers
        )
        tasks.append(
            asyncio.to_thread(client.encode_image, image_datas, normalize, n_dims)
        )

    if client.library_name not in EMBEDDING_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not an embedding model.",
        )

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    if texts is not None:
        text_embeddings = results[0]
        response["text_embeddings"] = [
            embedding.tolist() for embedding in text_embeddings
        ]

    if images is not None:
        # Whether it's index 1 or 0 depends on whether texts were provided
        image_embeddings = results[-1]
        response["image_embeddings"] = [
            embedding.tolist() for embedding in image_embeddings
        ]

    response["processing_time_ms"] = (end - start) * 1000

    return EmbeddingResponse(**response)


@app.post("/classify_image", response_model=ImageClassificationResponse)
async def classify_image(
    request: ImageClassificationRequest,
) -> ImageClassificationResponse:
    model_name = request.name
    images = request.image
    image_download_headers = request.image_download_headers

    client = client_from_cache(model_name, None)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not loaded. Load the model first using /load_model on the model server.",
        )

    if isinstance(images, str):
        images = [images]

    image_data = client.load_images_parallel(
        images, image_download_headers=image_download_headers
    )

    if image_data is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please provide a valid image URL or a base64 encoded image.",
        )

    if "image" not in client.modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} does not support image inference.",
        )

    if client.library_name not in CLASSIFICATION_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not a classification model.",
        )

    start = time.perf_counter()
    classifications = client.classify_image(image_data)
    end = time.perf_counter()

    return ImageClassificationResponse(
        probabilities=classifications,
        processing_time_ms=(end - start) * 1000,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    triton_metrics = TRITON_CLIENT.get_inference_statistics(as_json=True)
    triton_metrics["modelStats"] = triton_metrics.get("model_stats", [])
    return MetricsResponse(**triton_metrics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8686)
