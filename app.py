from fastapi import FastAPI, Request, HTTPException
import time
import os
import asyncio
from inference.request_models import (
    InferenceRequest,
    TextInferenceRequest,
    ImageInferenceRequest,
    GenericModelRequest,
    SentenceTransformerModelRequest,
    OpenCLIPModelRequest,
)
from inference.triton_open_clip.clip_model import TritonCLIPClient
from inference.triton_sentence_transformers.sentence_transformer_model import (
    TritonSentenceTransformersClient,
)
from inference.model_cache import LRUModelCache
from inference.common import get_model_name, delete_model_from_repo
import tritonclient.grpc as grpcclient
from threading import Lock

from typing import Union

TRITON_GRPC_URL = "localhost:8001"
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
TRITON_MODEL_REPOSITORY_PATH = "model_repository"

app = FastAPI()

# Model cache and lock
MODEL_CACHE = LRUModelCache(capacity=5)
MODEL_CACHE_LOCK = Lock()


def client_from_cache(
    model_name: str, pretrained: Union[str, None]
) -> Union[TritonCLIPClient, TritonSentenceTransformersClient, None]:
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
        client = TritonSentenceTransformersClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    if TRITON_CLIENT.is_model_ready(
        nice_model_name + "_text_encoder"
    ) and TRITON_CLIENT.is_model_ready(nice_model_name + "_image_encoder"):
        # if the model is ready, create a client for it
        # the model name must be split into text and image encoders for CLIP
        client = TritonCLIPClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    # if the model isn't ready, and the worker doesn't have it in cache, return None
    return None


@app.get("/health")
async def health():
    return {"message": "The server is running."}


@app.post("/load_clip_model")
async def load_clip_model(request: OpenCLIPModelRequest):
    model_name = request.model_name
    pretrained = request.pretrained
    cache_key = (model_name, pretrained)

    # Ensure thread safety when accessing the model cache
    with MODEL_CACHE_LOCK:
        client = MODEL_CACHE.get(cache_key)
        if client is None:
            client = TritonCLIPClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                pretrained=pretrained,
                triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            )
            MODEL_CACHE.put(cache_key, client)
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} loaded successfully."
            }
        else:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} is already loaded."
            }


@app.post("/load_sentence_transformer_model")
async def load_sentence_transformer_model(request: SentenceTransformerModelRequest):
    model_name = request.model_name

    cache_key = (model_name, None)

    # Ensure thread safety when accessing the model cache
    with MODEL_CACHE_LOCK:
        client = MODEL_CACHE.get(cache_key)
        if client is None:
            client = TritonSentenceTransformersClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            )
            MODEL_CACHE.put(cache_key, client)
            return {"message": f"Model {model_name} loaded successfully."}
        else:
            return {"message": f"Model {model_name} is already loaded."}


@app.post("/unload_model")
async def unload_model(request: GenericModelRequest):
    model_name = request.model_name
    pretrained = request.pretrained

    cache_key = (model_name, pretrained)

    with MODEL_CACHE_LOCK:
        client = MODEL_CACHE.get(cache_key)
        if client is not None:
            client = MODEL_CACHE.remove(cache_key)
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} unloaded successfully."
            }
        else:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} is not loaded."
            }


@app.post("/delete_model")
async def delete_model(request: GenericModelRequest):
    model_name = request.model_name
    pretrained = request.pretrained

    cache_key = (model_name, pretrained)

    with MODEL_CACHE_LOCK:
        delete_model_from_repo(model_name, pretrained, TRITON_MODEL_REPOSITORY_PATH)
        client = MODEL_CACHE.get(cache_key)
        if client is not None:
            client = MODEL_CACHE.remove(cache_key)

        return {
            "message": f"Model {model_name} with checkpoint {pretrained} deleted successfully."
        }


@app.get("/loaded_models")
async def loaded_models():
    model_repo_information = TRITON_CLIENT.get_model_repository_index(as_json=True)
    loaded_models = []
    for model in model_repo_information["models"]:
        if "state" in model and model["state"] == "UNAVAILABLE":
            continue
        loaded_models.append(model["name"])
    return {"models": loaded_models}


@app.get("/repository_models")
async def repository_models():
    model_repo_information = TRITON_CLIENT.get_model_repository_index(as_json=True)
    repository_models = []
    for model in model_repo_information["models"]:
        model_data = {"name": model["name"]}
        print(model)
        if "state" in model:
            model_data["state"] = model["state"]

        repository_models.append(model_data)
    return {"models": repository_models}


@app.post("/infer_text")
async def infer_text(request: TextInferenceRequest):
    model_name = request.model_name
    pretrained = request.pretrained
    text = request.text
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model.",
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

    return {"embedding": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer_image")
async def infer_text(request: ImageInferenceRequest):
    model_name = request.model_name
    pretrained = request.pretrained
    image = request.image
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model.",
        )

    image_data = client.load_image(image)

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
    return {"embedding": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer")
async def infer(request: InferenceRequest):
    model_name = request.model_name
    pretrained = request.pretrained
    texts = request.text
    images = request.image
    normalize = request.normalize

    client = client_from_cache(model_name, pretrained)

    if client is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with checkpoint {pretrained} is not loaded. Load the model first using /load_model.",
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
        # image_datas = [client.load_image(image) for image in images]
        image_datas = client.load_images_parallel(images)
        tasks.append(asyncio.to_thread(client.encode_image, image_datas, normalize))

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

    response["processingTimeMs"] = (end - start) * 1000

    return response


@app.get("/metrics")
async def metrics():
    triton_metrics = TRITON_CLIENT.get_inference_statistics(as_json=True)
    return triton_metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8686)
