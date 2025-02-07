from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import os
import asyncio
from ingrain_inference.api_models.request_models import (
    GenericModelRequest,
    SentenceTransformerModelRequest,
    OpenCLIPModelRequest,
    TimmModelRequest,
    DownloadCustomModelRequest,
)
from ingrain_inference.api_models.response_models import (
    GenericMessageResponse,
    LoadedModelResponse,
    RepositoryModelResponse,
)
from ingrain_inference.inference.triton_open_clip.clip_model import (
    TritonCLIPModelClient,
)
from ingrain_inference.inference.triton_sentence_transformers.sentence_transformer_model import (
    TritonSentenceTransformersModelClient,
)
from ingrain_inference.inference.triton_timm.timm_model import TritonTimmModelClient

from ingrain_inference.inference.model_cache import LRUModelCache
from ingrain_inference.inference.common import get_model_name, delete_model_from_repo
from ingrain_inference.inference.custom_model_utils import (
    download_custom_open_clip_model,
    download_custom_sentence_transformers_model,
    download_custom_timm_model,
)
import tritonclient.grpc as grpcclient
from threading import Lock

from typing import Union, Literal

TRITON_GRPC_URL = "localhost:8001"
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
TRITON_MODEL_REPOSITORY_PATH = "model_repository"
CUSTOM_MODEL_DIR = "custom_model_files"

# faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

app = FastAPI()

# Model cache and lock
MODEL_CACHE = LRUModelCache(capacity=5)
MODEL_CACHE_LOCK = Lock()


def get_model_creation_client(
    model_name: str,
    pretrained: Union[str, None],
    model_library: Literal["open_clip", "sentence_transformers", "timm"],
) -> Union[
    TritonCLIPModelClient,
    TritonTimmModelClient,
    TritonSentenceTransformersModelClient,
    None,
]:
    nice_model_name = get_model_name(model_name, pretrained)

    cache_key = (model_name, pretrained)

    if (
        TRITON_CLIENT.is_model_ready(nice_model_name)
        and model_library == "sentence_transformers"
    ):
        # if the model is ready, create a client for it
        # the model name is used directly for sentence transformers
        client = TritonSentenceTransformersModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    if TRITON_CLIENT.is_model_ready(nice_model_name) and model_library == "timm":
        # if the model is ready, create a client for it
        # the model name is used directly for timm models
        client = TritonTimmModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
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
        client = TritonCLIPModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return client

    return None


@app.middleware("http")
async def route_timeout_middleware(request: Request, call_next):
    if request.url.path in {
        "/load_clip_model",
        "/load_sentence_transformer_model",
        "/load_timm_model",
    }:
        try:
            return await asyncio.wait_for(call_next(request), timeout=600)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "Request timed out"}, status_code=504)
    else:
        return await call_next(request)


@app.get("/health")
async def health() -> GenericMessageResponse:
    return {"message": "The model server is running."}


@app.post("/load_clip_model")
async def load_clip_model(request: OpenCLIPModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained
    cache_key = (model_name, pretrained)

    client = get_model_creation_client(
        model_name, pretrained, model_library="open_clip"
    )
    if client is None:
        client = TritonCLIPModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return {
            "message": f"Model {model_name} with checkpoint {pretrained} loaded successfully."
        }
    else:
        client.load()
        return {
            "message": f"Model {model_name} with checkpoint {pretrained} is already loaded."
        }


@app.post("/load_sentence_transformer_model")
async def load_sentence_transformer_model(
    request: SentenceTransformerModelRequest,
) -> GenericMessageResponse:
    model_name = request.name

    cache_key = (model_name, None)
    client = get_model_creation_client(
        model_name, None, model_library="sentence_transformers"
    )
    if client is None:
        client = TritonSentenceTransformersModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return {"message": f"Model {model_name} loaded successfully."}
    else:
        client.load()
        return {"message": f"Model {model_name} is already loaded."}


@app.post("/load_timm_model")
async def load_timm_model(request: TimmModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    cache_key = (model_name, pretrained)

    client = get_model_creation_client(model_name, pretrained, model_library="timm")
    if client is None:
        client = TritonTimmModelClient(
            triton_grpc_url=TRITON_GRPC_URL,
            model=model_name,
            pretrained=pretrained,
            triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
            custom_model_dir=CUSTOM_MODEL_DIR,
        )
        with MODEL_CACHE_LOCK:
            MODEL_CACHE.put(cache_key, client)
        return {"message": f"Model {model_name} loaded successfully."}
    else:
        client.load()
        return {"message": f"Model {model_name} is already loaded."}


@app.post("/unload_model")
async def unload_model(request: GenericModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    cache_key = (model_name, pretrained)
    original_cache_size = len(MODEL_CACHE)
    with MODEL_CACHE_LOCK:
        MODEL_CACHE.remove(cache_key)
        if len(MODEL_CACHE) < original_cache_size:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} unloaded successfully."
            }
        else:
            return {
                "message": f"Model {model_name} with checkpoint {pretrained} is not loaded."
            }


@app.delete("/delete_model")
async def delete_model(request: GenericModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    cache_key = (model_name, pretrained)

    with MODEL_CACHE_LOCK:
        client = MODEL_CACHE.get(cache_key)
        if client is not None:
            client = MODEL_CACHE.remove(cache_key)
            delete_model_from_repo(model_name, pretrained, TRITON_MODEL_REPOSITORY_PATH)
        return {
            "message": f"Model {model_name} with checkpoint {pretrained} deleted successfully."
        }


@app.get("/loaded_models")
async def loaded_models() -> LoadedModelResponse:
    model_repo_information = TRITON_CLIENT.get_model_repository_index(as_json=True)
    loaded_models = []
    for model in model_repo_information.get("models", []):
        if "state" in model and model["state"] == "UNAVAILABLE":
            continue
        loaded_models.append(model["name"])
    return {"models": loaded_models}


@app.get("/repository_models")
async def repository_models() -> RepositoryModelResponse:
    model_repo_information = TRITON_CLIENT.get_model_repository_index(as_json=True)
    repository_models = []
    for model in model_repo_information.get("models", []):
        model_data = {"name": model["name"]}
        if "state" in model:
            model_data["state"] = model["state"]

        repository_models.append(model_data)
    return {"models": repository_models}


@app.post("/download_custom_model")
async def download_custom_model(
    request: DownloadCustomModelRequest,
) -> GenericMessageResponse:
    model_library = request.library
    model_name = request.pretrained_name
    model_url = request.safetensors_url

    if model_library == "open_clip":
        try:
            download_custom_open_clip_model(
                CUSTOM_MODEL_DIR,
                model_name,
                model_url,
                mode=request.mode,
                mean=request.mean,
                std=request.std,
                interpolation=request.interpolation,
                resize_mode=request.resize_mode,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error in downloading custom model, original error: {str(e)}",
            )

    if model_library == "sentence_transformers":
        try:
            download_custom_sentence_transformers_model(
                CUSTOM_MODEL_DIR,
                model_name,
                model_url,
                config_json_url=request.config_json_url,
                tokenizer_json_url=request.tokenizer_json_url,
                tokenizer_config_json_url=request.tokenizer_config_json_url,
                vocab_txt_url=request.vocab_txt_url,
                special_tokens_map_json_url=request.special_tokens_map_json_url,
                pooling_config_json_url=request.pooling_config_json_url,
                sentence_bert_config_json_url=request.sentence_bert_config_json_url,
                modules_json_url=request.modules_json_url,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error in downloading custom model, original error: {str(e)}",
            )

    if model_library == "timm":
        try:
            download_custom_timm_model(
                CUSTOM_MODEL_DIR,
                model_name,
                model_url,
                num_classes=request.num_classes,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error in downloading custom model, original error: {str(e)}",
            )

    return {"message": f"Custom model {model_name} downloaded successfully."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8687)
