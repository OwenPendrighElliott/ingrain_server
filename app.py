from fastapi import FastAPI, Request, HTTPException
import time
import os
import asyncio
from inference.triton_open_clip.clip_model import TritonCLIPClient
from inference.triton_sentence_transformers.sentence_transformer_model import (
    TritonSentenceTransformersClient,
)
import tritonclient.grpc as grpcclient
from threading import Lock

from typing import Dict, Tuple, Union

TRITON_GRPC_URL = "localhost:8001"

TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)

app = FastAPI()

# Model cache and lock
MODEL_CACHE: Dict[Tuple[str, Union[str, None]], TritonCLIPClient] = {}
MODEL_CACHE_LOCK = Lock()

def client_from_cache(
    model_name: str, pretrained: Union[str, None]
) -> Union[TritonCLIPClient, TritonSentenceTransformersClient, None]:
    cache_key = (model_name, pretrained)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    nice_model_name = model_name.replace("/", "_")
    if pretrained is not None:
        nice_model_name += f"_{pretrained}"

    if TRITON_CLIENT.is_model_ready(nice_model_name):
        try:
            client = TritonSentenceTransformersClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                triton_model_repository_path="model_repository",
            )
            MODEL_CACHE[cache_key] = client
            return client
        except:
            pass
        
        try:
            client = TritonCLIPClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                pretrained=pretrained,
                triton_model_repository_path="model_repository",
            )
            MODEL_CACHE[cache_key] = client
            return client
        except:
            pass

    return None


@app.get("/health")
async def read_root():
    return {"message": "The server is running."}


@app.post("/load_clip_model")
async def load_clip_model(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    pretrained = data.get("pretrained", None)
    cache_key = (model_name, pretrained)

    # Ensure thread safety when accessing the model cache
    with MODEL_CACHE_LOCK:
        if cache_key not in MODEL_CACHE:
            client = TritonCLIPClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                pretrained=pretrained,
                triton_model_repository_path="model_repository",
            )
            MODEL_CACHE[cache_key] = client
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} loaded successfully."
            }
        else:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} is already loaded."
            }


@app.post("/load_sentence_transformer_model")
async def load_sentence_transformer_model(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    cache_key = (model_name, None)

    # Ensure thread safety when accessing the model cache
    with MODEL_CACHE_LOCK:
        if cache_key not in MODEL_CACHE:
            client = TritonSentenceTransformersClient(
                triton_grpc_url=TRITON_GRPC_URL,
                model=model_name,
                triton_model_repository_path="model_repository",
            )
            MODEL_CACHE[cache_key] = client
            return {"message": f"Model {model_name} loaded successfully."}
        else:
            return {"message": f"Model {model_name} is already loaded."}


@app.post("/unload_model")
async def unload_model(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    pretrained = data.get("pretrained", None)
    cache_key = (model_name, pretrained)

    # Ensure thread safety when accessing the model cache
    with MODEL_CACHE_LOCK:
        if cache_key in MODEL_CACHE:
            client = MODEL_CACHE.pop(cache_key)
            client.unload()
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} unloaded successfully."
            }
        else:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} is not loaded."
            }


@app.get("/loaded_models")
async def loaded_models():
    with MODEL_CACHE_LOCK:
        return {
            "models": [
                {"model_name": model_name, "pretrained": pretrained}
                for model_name, pretrained in MODEL_CACHE.keys()
            ]
        }


@app.post("/infer_text")
async def infer_text(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    pretrained = data.get("pretrained", None)
    text = data["text"]
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
    embedding = client.encode_text(text)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]

    return {"embedding": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer_image")
async def infer_text(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    pretrained = data.get("pretrained", None)
    image = data["image"]

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
    embedding = client.encode_image(image_data)
    end = time.perf_counter()

    embedding_list = [e.tolist() for e in embedding]
    return {"embedding": embedding_list, "processingTimeMs": (end - start) * 1000}


@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    pretrained = data.get("pretrained", None)
    texts = data.get("texts", None)
    images = data.get("images", None)

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
        tasks.append(asyncio.to_thread(client.encode_text, texts))

    if images is not None:
        if "image" not in client.modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} with checkpoint {pretrained} does not support image inference.",
            )
        image_datas = [client.load_image(image) for image in images]
        tasks.append(asyncio.to_thread(client.encode_image, image_datas))

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
