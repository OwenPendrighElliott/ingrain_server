from fastapi import FastAPI, HTTPException
import time
import asyncio
from inference.api_models.request_models import (
    InferenceRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
)
from inference.api_models.response_models import (
    InferenceResponse,
    TextInferenceResponse,
    ImageInferenceResponse,
    GenericMessageResponse,
    MetricsResponse,
)
from inference.triton_open_clip.clip_model import TritonCLIPInferenceClient
from inference.triton_sentence_transformers.sentence_transformer_model import (
    TritonSentenceTransformersInferenceClient,
)
from inference.common import get_model_name
from inference.model_cache import LRUModelCache
from threading import Lock
import tritonclient.grpc as grpcclient
from typing import Union

TRITON_GRPC_URL = "localhost:8001"
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)

app = FastAPI()

# Model cache and lock
MODEL_CACHE = LRUModelCache(capacity=5)
MODEL_CACHE_LOCK = Lock()


def client_from_cache(
    model_name: str, pretrained: Union[str, None]
) -> Union[TritonCLIPInferenceClient, TritonSentenceTransformersInferenceClient, None]:
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

    # if the model isn't in this workers cache, check if it's ready
    if TRITON_CLIENT.is_model_ready(nice_model_name):
        # if the model is ready, create a client for it
        # the model name is used directly for sentence transformers
        client = TritonSentenceTransformersInferenceClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    if TRITON_CLIENT.is_model_ready(
        nice_model_name + "_text_encoder"
    ) and TRITON_CLIENT.is_model_ready(nice_model_name + "_image_encoder"):
        # if the model is ready, create a client for it
        # the model name must be split into text and image encoders for CLIP
        client = TritonCLIPInferenceClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    # if the model isn't ready, and the worker doesn't have it in cache, return None
    return None


@app.get("/health")
async def health() -> GenericMessageResponse:
    return {"message": "The inference server is running."}


@app.post("/infer_text")
async def infer_text(request: TextInferenceRequest) -> TextInferenceResponse:
    model_name = request.name
    pretrained = request.pretrained
    text = request.text
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model on the model server.",
        )

    if "text" not in client.modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} does not support text inference.",
        )

    start = time.perf_counter()
    embedding = client.encode_text(text, normalize=normalize)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]

    return {"embeddings": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer_image")
async def infer_image(request: ImageInferenceRequest) -> ImageInferenceResponse:
    model_name = request.name
    pretrained = request.pretrained
    images = request.image
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model on the model server.",
        )

    if isinstance(images, str):
        images = [images]

    image_data = client.load_images_parallel(images)

    if image_data is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please provide a valid image URL or a base64 encoded image.",
        )

    if "image" not in client.modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} does not support image inference.",
        )

    start = time.perf_counter()
    embedding = client.encode_image(image_data, normalize=normalize)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]
    return {"embeddings": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer")
async def infer(request: InferenceRequest) -> InferenceResponse:
    model_name = request.name
    pretrained = request.pretrained
    texts = request.text
    images = request.image
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)

    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model on the model server.",
        )

    response = {}
    tasks = []

    if texts is not None:
        if "text" not in client.modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} with checkpoint {pretrained} does not support text inference.",
            )
        if isinstance(texts, str):
            texts = [texts]
        tasks.append(asyncio.to_thread(client.encode_text, texts, normalize))

    if images is not None:
        if "image" not in client.modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} with checkpoint {pretrained} does not support image inference.",
            )
        if isinstance(images, str):
            images = [images]
        image_datas = client.load_images_parallel(images)
        tasks.append(asyncio.to_thread(client.encode_image, image_datas, normalize))

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    if texts is not None:
        text_embeddings = results[0]
        response["textEmbeddings"] = [
            embedding.tolist() for embedding in text_embeddings
        ]

    if images is not None:
        # Whether it's index 1 or 0 depends on whether texts were provided
        image_embeddings = results[-1]
        response["imageEmbeddings"] = [
            embedding.tolist() for embedding in image_embeddings
        ]

    response["processingTimeMs"] = (end - start) * 1000

    return response


@app.get("/metrics")
async def metrics() -> MetricsResponse:
    triton_metrics = TRITON_CLIENT.get_inference_statistics(as_json=True)
    triton_metrics["modelStats"] = triton_metrics["model_stats"]
    return triton_metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8686)
