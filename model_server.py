from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import os
import asyncio
from ingrain_models.api_models.request_models import (
    GenericModelRequest,
    SentenceTransformerModelRequest,
    OpenCLIPModelRequest,
    TimmModelRequest,
    DownloadCustomModelRequest,
)
from ingrain_models.api_models.response_models import (
    GenericMessageResponse,
    LoadedModelResponse,
    RepositoryModelResponse,
)
from ingrain_models.models.model_client import TritonModelLoadingClient

from ingrain_common.common import delete_model_from_repo
from ingrain_models.models.custom_model_utils import (
    download_custom_open_clip_model,
    download_custom_sentence_transformers_model,
    download_custom_timm_model,
)
import tritonclient.grpc as grpcclient
from threading import Lock

from typing import Union, Literal, Tuple, Dict

TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 5))
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
TRITON_MODEL_REPOSITORY_PATH = "model_repository"
CUSTOM_MODEL_DIR = "custom_model_files"

# faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

app = FastAPI()

MODEL_CACHE: Dict[
    Tuple[str, str | None],
    Tuple[
        str,
        str | None,
        Literal["open_clip", "sentence_transformers", "timm"],
        Literal["loaded", "unloaded"],
    ],
] = {}
CACHE_LOCK = Lock()


def count_loaded_models() -> int:
    """Count the number of currently loaded models in the Triton server."""
    model_repo_information: dict = TRITON_CLIENT.get_model_repository_index(
        as_json=True
    )
    models = [
        model
        for model in model_repo_information.get("models", [])
        if "state" in model and model["state"] == "READY"
    ]
    return len(models), models


def update_model_cache(
    model_name: str,
    pretrained: Union[str, None],
    model_library: Literal["open_clip", "sentence_transformers", "timm"] | None,
    operation: Literal["load", "unload", "delete"],
) -> None:
    cache_key = (model_name, pretrained)
    with CACHE_LOCK:
        if operation == "load":
            MODEL_CACHE[cache_key] = (model_name, pretrained, model_library, "loaded")
        elif operation == "unload":
            MODEL_CACHE[cache_key] = (model_name, pretrained, model_library, "unloaded")
        elif operation == "delete":
            if cache_key in MODEL_CACHE:
                del MODEL_CACHE[cache_key]

    print(MODEL_CACHE)


def get_model_creation_client(
    model_name: str,
    pretrained: Union[str, None],
    model_library: Literal["open_clip", "sentence_transformers", "timm"],
) -> TritonModelLoadingClient:
    return TritonModelLoadingClient(
        triton_grpc_url=TRITON_GRPC_URL,
        model=model_name,
        pretrained=pretrained,
        library_name=model_library,
        triton_model_repository_path=TRITON_MODEL_REPOSITORY_PATH,
        custom_model_dir=CUSTOM_MODEL_DIR,
    )


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

    try:
        client = get_model_creation_client(
            model_name, pretrained, model_library="open_clip"
        )
        if not client.is_in_repository():
            client.create_triton_model()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error in creating model client: {str(e)}",
        )

    if not client.is_ready():
        loaded_models, models = count_loaded_models()
        if loaded_models >= MAX_LOADED_MODELS:
            raise HTTPException(
                status_code=503,
                detail=f"Maximum number of loaded models ({MAX_LOADED_MODELS}) reached. Please unload a model before loading a new one. Currently loaded models: {', '.join(model['name'] for model in models)}",
            )
    else:
        update_model_cache(model_name, pretrained, "open_clip", operation="load")
        return {
            "message": f"Model {model_name} with checkpoint {pretrained} is already loaded."
        }

    try:
        client.load()
        update_model_cache(model_name, pretrained, "open_clip", operation="load")
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model {model_name} with checkpoint {pretrained}: {str(e)}",
        )

    return {
        "message": f"Model {model_name} with checkpoint {pretrained} loaded successfully."
    }


@app.post("/load_sentence_transformer_model")
async def load_sentence_transformer_model(
    request: SentenceTransformerModelRequest,
) -> GenericMessageResponse:
    model_name = request.name

    try:
        client = get_model_creation_client(
            model_name, None, model_library="sentence_transformers"
        )
        if not client.is_in_repository():
            client.create_triton_model()
    except ValueError as e:
        print(e)
        raise HTTPException(
            status_code=400,
            detail=f"Error in creating model client: {str(e)}",
        )

    if not client.is_ready():
        loaded_models, models = count_loaded_models()
        print(loaded_models, models)
        if loaded_models >= MAX_LOADED_MODELS:
            raise HTTPException(
                status_code=503,
                detail=f"Maximum number of loaded models ({MAX_LOADED_MODELS}) reached. Please unload a model before loading a new one. Currently loaded models: {', '.join(model['name'] for model in models)}",
            )
    else:
        update_model_cache(model_name, None, "open_clip", operation="load")
        return {
            "message": f"Model {model_name} is already loaded."
        }

    try:
        client.load()
        update_model_cache(model_name, None, "open_clip", operation="load")
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model {model_name}: {str(e)}",
        )

    return {"message": f"Model {model_name} loaded successfully."}


@app.post("/load_timm_model")
async def load_timm_model(request: TimmModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    try:
        client = get_model_creation_client(model_name, pretrained, model_library="timm")
        if not client.is_in_repository():
            client.create_triton_model()
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error in creating model client: {str(e)}",
        )

    if not client.is_ready():
        loaded_models, models = count_loaded_models()
        if loaded_models >= MAX_LOADED_MODELS:
            raise HTTPException(
                status_code=503,
                detail=f"Maximum number of loaded models ({MAX_LOADED_MODELS}) reached. Please unload a model before loading a new one. Currently loaded models: {', '.join(model['name'] for model in models)}",
            )
    else:
        update_model_cache(model_name, pretrained, "open_clip", operation="load")
        return {
            "message": f"Model {model_name} with checkpoint {pretrained} is already loaded."
        }

    try:
        client.load()
        update_model_cache(model_name, pretrained, "open_clip", operation="load")
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model {model_name} with checkpoint {pretrained}: {str(e)}",
        )

    return {
        "message": f"Model {model_name} with checkpoint {pretrained} loaded successfully."
    }


@app.post("/unload_model")
async def unload_model(request: GenericModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    print(model_name, pretrained)
    print(MODEL_CACHE)
    print(TRITON_CLIENT.get_model_repository_index())

    with CACHE_LOCK:
        _, _, model_library, _ = MODEL_CACHE.get(
            (model_name, pretrained), (None, None, None, None)
        )
        if model_library is None:
            update_model_cache(model_name, pretrained, model_library, operation="unload")
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} with checkpoint {pretrained} is not loaded.",
            )

    client = get_model_creation_client(
        model_name, pretrained, model_library=model_library
    )

    try:
        client.unload()
        update_model_cache(model_name, pretrained, model_library, operation="unload")
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error unloading model {model_name} with checkpoint {pretrained}: {str(e)}",
        )
    return {
        "message": f"Model {model_name} with checkpoint {pretrained} unloaded successfully."
    }


@app.delete("/delete_model")
async def delete_model(request: GenericModelRequest) -> GenericMessageResponse:
    model_name = request.name
    pretrained = request.pretrained

    with CACHE_LOCK:
        _, _, model_library, state = MODEL_CACHE.get(
            (model_name, pretrained), (None, None, None, None)
        )

    if state is None:
        update_model_cache(model_name, pretrained, model_library, operation="delete")
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} with checkpoint {pretrained} does not exist.",
        )

    client = get_model_creation_client(
        model_name, pretrained, model_library=model_library
    )

    try:
        client.unload()
        delete_model_from_repo(model_name, pretrained, TRITON_MODEL_REPOSITORY_PATH)
        update_model_cache(model_name, pretrained, model_library, operation="delete")
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting model {model_name} with checkpoint {pretrained}: {str(e)}",
        )
    return {
        "message": f"Model {model_name} with checkpoint {pretrained} deleted successfully."
    }


@app.get("/loaded_models")
async def loaded_models() -> LoadedModelResponse:
    model_repo_information: dict = TRITON_CLIENT.get_model_repository_index(
        as_json=True
    )
    loaded_models = []
    for model in model_repo_information.get("models", []):
        if "state" in model and model["state"] == "UNAVAILABLE":
            continue
        loaded_models.append(model["name"])
    return {"models": loaded_models}


@app.get("/repository_models")
async def repository_models() -> RepositoryModelResponse:
    model_repo_information: dict = TRITON_CLIENT.get_model_repository_index(
        as_json=True
    )
    repository_models = []
    for model in model_repo_information.get("models", []):
        model_data = {"name": model["name"]}
        model_data["state"] = model.get("state", "NOT READY")

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
