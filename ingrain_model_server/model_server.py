from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import os
import asyncio
from ingrain_models.api_models.request_models import (
    LoadModelRequest,
    UnloadModelRequest,
    DownloadCustomModelRequest,
    ModelMetadataRequest,
)
from ingrain_models.api_models.response_models import (
    GenericMessageResponse,
    LoadedModelResponse,
    RepositoryModelResponse,
    ModelEmbeddingDimsResponse,
    ModelClassificationLabelsResponse,
)
from ingrain_models.models.model_client import TritonModelLoadingClient

from ingrain_common.common import (
    delete_model_from_repo,
    get_library_name,
    get_model_name,
    get_text_image_model_names,
    validate_env_vars,
)
from ingrain_models.models.custom_model_utils import (
    download_custom_open_clip_model,
    download_custom_sentence_transformers_model,
    download_custom_timm_model,
)
import tritonclient.grpc as grpcclient

from typing import Union, Literal

TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 5))
TRITON_CLIENT = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
TRITON_MODEL_REPOSITORY_PATH = "model_repository"
CUSTOM_MODEL_DIR = "custom_model_files"

EMBEDDING_MODEL_LIBRARIES = ["open_clip", "sentence_transformers"]
CLASSIFICATION_MODEL_LIBRARIES = ["timm"]

# faster model downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)

app = FastAPI()


validate_env_vars()


def count_loaded_models() -> tuple[int, list[str]]:
    """Count the number of currently loaded models in the Triton server."""
    model_repo_information: dict = TRITON_CLIENT.get_model_repository_index(
        as_json=True
    )
    models = [
        model["name"].replace("_text_encoder", "").replace("_image_encoder", "")
        for model in model_repo_information.get("models", [])
        if "state" in model and model["state"] == "READY"
    ]
    models = list(set(models))
    return len(models), models


def get_library_name_for_model(model_name: str, pretrained: Union[str, None]) -> str:
    triton_model_name = get_model_name(model_name, pretrained)
    try:
        library_name = get_library_name(
            os.path.join(TRITON_MODEL_REPOSITORY_PATH, triton_model_name)
        )
    except FileNotFoundError:
        try:
            text_triton_name, _ = get_text_image_model_names(model_name, pretrained)
            library_name = get_library_name(
                os.path.join(TRITON_MODEL_REPOSITORY_PATH, text_triton_name)
            )
        except FileNotFoundError:
            library_name = None

    return library_name


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
    if request.url.path == "/load_model":
        try:
            return await asyncio.wait_for(call_next(request), timeout=600)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "Request timed out"}, status_code=504)
    else:
        return await call_next(request)


@app.get("/health", response_model=GenericMessageResponse)
async def health() -> GenericMessageResponse:
    return GenericMessageResponse(message="The model server is running.")


@app.post("/load_model", response_model=GenericMessageResponse)
async def load_model(request: LoadModelRequest) -> GenericMessageResponse:
    model_name = request.name
    library = request.library

    try:
        client = get_model_creation_client(model_name, None, model_library=library)
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
                status_code=400,
                detail=f"Maximum number of loaded models ({MAX_LOADED_MODELS}) reached. Please unload a model before loading a new one. Currently loaded models: {', '.join(models)}",
            )
    else:
        return GenericMessageResponse(message=f"Model {model_name} is already loaded.")

    try:
        client.load()
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model {model_name}: {str(e)}\nIt is possible that triton is unavailable, the model is not compatible, or that the incorrect model repository directory is being use.",
        )

    return GenericMessageResponse(message=f"Model {model_name} loaded successfully.")


@app.post("/unload_model", response_model=GenericMessageResponse)
async def unload_model(request: UnloadModelRequest) -> GenericMessageResponse:
    model_name = request.name

    model_library = get_library_name_for_model(model_name, None)
    triton_model_name = get_model_name(model_name, None)
    if model_library == "open_clip":
        text_triton_name, image_triton_name = get_text_image_model_names(
            model_name, None
        )
        ready = TRITON_CLIENT.is_model_ready(
            text_triton_name
        ) and TRITON_CLIENT.is_model_ready(image_triton_name)
    else:
        ready = TRITON_CLIENT.is_model_ready(triton_model_name)

    if model_library and not ready:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} is not loaded.",
        )
    if not model_library:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} does not exist in the repository.",
        )

    client = get_model_creation_client(model_name, None, model_library=model_library)

    try:
        client.unload()
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error unloading model {model_name}: {str(e)}\nIt is possible that triton is unavailable, the model is not compatible, or that the incorrect model repository directory is being use",
        )
    return GenericMessageResponse(message=f"Model {model_name} unloaded successfully.")


@app.delete("/delete_model", response_model=GenericMessageResponse)
async def delete_model(request: UnloadModelRequest) -> GenericMessageResponse:
    model_name = request.name

    model_library = get_library_name_for_model(model_name, None)

    if not model_library:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} does not exist.",
        )

    client = get_model_creation_client(model_name, None, model_library=model_library)

    try:
        client.unload()
        delete_model_from_repo(model_name, None, TRITON_MODEL_REPOSITORY_PATH)
    except grpcclient.InferenceServerException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting model {model_name}: {str(e)}\nIt is possible that triton is unavailable, the model is not compatible, or that the incorrect model repository directory is being use",
        )
    return GenericMessageResponse(message=f"Model {model_name} deleted successfully.")


@app.get("/loaded_models", response_model=LoadedModelResponse)
async def loaded_models() -> LoadedModelResponse:
    _, loaded_models = count_loaded_models()
    return LoadedModelResponse(models=loaded_models)


@app.get("/repository_models", response_model=RepositoryModelResponse)
async def repository_models() -> RepositoryModelResponse:
    model_repo_information: dict = TRITON_CLIENT.get_model_repository_index(
        as_json=True
    )
    repository_models = []
    for model in model_repo_information.get("models", []):
        model_data = {"name": model["name"]}
        model_data["state"] = model.get("state", "NOT READY")

        repository_models.append(model_data)
    return RepositoryModelResponse(models=repository_models)


@app.post("/download_custom_model", response_model=GenericMessageResponse)
async def download_custom_model(
    request: DownloadCustomModelRequest,
) -> GenericMessageResponse:
    model_library = request.library
    model_name = request.name
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

    return GenericMessageResponse(
        message=f"Custom model {model_name} downloaded successfully."
    )


@app.post("/model_embedding_size", response_model=ModelEmbeddingDimsResponse)
async def model_embedding_size(
    request: ModelMetadataRequest,
) -> ModelEmbeddingDimsResponse:
    model_name = request.name
    model_library = get_library_name_for_model(model_name, None)

    if not model_library:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} does not exist in the repository.",
        )

    if model_library not in EMBEDDING_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not an embedding model.",
        )

    # for open_clip models, get the text encoder embedding size
    if model_library == "open_clip":
        model_name, _ = get_text_image_model_names(model_name, None)

    metadata = TRITON_CLIENT.get_model_metadata(
        model_name=get_model_name(model_name, None), as_json=True
    )
    return ModelEmbeddingDimsResponse(
        embedding_size=metadata["outputs"][0]["shape"][-1]
    )


@app.post(
    "/model_classification_labels",
    response_model=ModelClassificationLabelsResponse,
)
async def model_classification_labels(
    request: ModelMetadataRequest,
) -> ModelClassificationLabelsResponse:
    model_name = request.name
    model_library = get_library_name_for_model(model_name, None)
    if not model_library:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} does not exist in the repository.",
        )

    if model_library not in CLASSIFICATION_MODEL_LIBRARIES:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} is not a classification model.",
        )

    labels_path = os.path.join(
        TRITON_MODEL_REPOSITORY_PATH, get_model_name(model_name, None), "classes.txt"
    )
    if not os.path.exists(labels_path):
        raise HTTPException(
            status_code=404,
            detail=f"Labels file for model {model_name} does not exist.",
        )

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return ModelClassificationLabelsResponse(labels=labels)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8687)
