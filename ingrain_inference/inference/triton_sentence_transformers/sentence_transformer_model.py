import os
import json
from sentence_transformers import SentenceTransformer
from ..model_client import TritonModelLoadingClient
from ..common import get_model_name, save_library_name, custom_model_exists
from .sentence_transformer_converting import (
    onnx_transformer_model,
    generate_text_sentence_transformer_config,
)

from typing import Union, List, Optional


def create_model(
    model_name: str,
    triton_model_repository_path: str,
    custom_model_dir: str,
):

    if custom_model_exists(custom_model_dir, model_name):
        with open(
            os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "r"
        ) as f:
            model_meta = json.load(f)
        if model_meta["model_type"] != "sentence_transformers":
            raise ValueError(
                f"The custom model {model_name} exists but it is not a sentence_transformers model."
            )

        model = SentenceTransformer(os.path.join(custom_model_dir, model_name))
    else:
        model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    friendly_name = get_model_name(model_name)

    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_name, "1"), exist_ok=True
    )

    text_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_name, "1", "model.onnx")
    )
    text_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_name, "config.pbtxt")
    )

    # exit early if the models and configs already exist
    if text_encoder_exists and text_config_exists:
        return friendly_name, tokenizer

    onnx_transformer_model(
        model,
        os.path.join(triton_model_repository_path, friendly_name, "1", "model.onnx"),
    )

    generate_text_sentence_transformer_config(
        os.path.join(triton_model_repository_path, friendly_name),
        friendly_name,
        model.get_sentence_embedding_dimension(),
    )

    save_library_name(
        os.path.join(triton_model_repository_path, friendly_name),
        "sentence_transformers",
    )

    with open(
        os.path.join(
            triton_model_repository_path,
            friendly_name,
            "sentence_transformer_config.json",
        ),
        "w+",
    ) as f:
        data = {"max_length": model._modules["0"].max_seq_length}
        json.dump(data, f)

    return friendly_name, tokenizer


class TritonSentenceTransformersModelClient(TritonModelLoadingClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        triton_model_repository_path: str,
        custom_model_dir: str,
    ):
        super().__init__(triton_grpc_url)

        self.model_name = get_model_name(model)

        if not self.triton_client.is_model_ready(self.model_name):
            _, _ = create_model(model, triton_model_repository_path, custom_model_dir)
            self.triton_client.load_model(self.model_name)

        self.modalities = {"text"}

    def load(self):
        self.triton_client.load_model(self.model_name)

    def unload(self):
        self.triton_client.unload_model(self.model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_name)
