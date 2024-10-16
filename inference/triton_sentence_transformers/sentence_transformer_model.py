import os
import numpy as np
import tritonclient.grpc as grpcclient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from ..model_client import TritonModelInferenceClient, TritonModelLoadingClient
from ..common import get_model_name
from .sentence_transformer_converting import (
    onnx_transformer_model,
    generate_text_sentence_transformer_config,
)

from typing import Union, List


def create_model(
    model_name: str,
    triton_model_repository_path: str,
):
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

    return friendly_name, tokenizer


class TritonSentenceTransformersInferenceClient(TritonModelInferenceClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
    ):
        super().__init__(triton_grpc_url)

        self.model_name = get_model_name(model)

        if not self.triton_client.is_model_ready(self.model_name):
            raise ValueError(f"Model {model} is not ready")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.modalities = {"text"}

    def encode_text(
        self, text: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        tokens = self.tokenizer(
            text, return_tensors="np", padding=True, truncation=True
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        input_ids_tensor = grpcclient.InferInput("input_ids", input_ids.shape, "INT64")
        attention_mask_tensor = grpcclient.InferInput(
            "attention_mask", attention_mask.shape, "INT64"
        )

        input_ids_tensor.set_data_from_numpy(input_ids)
        attention_mask_tensor.set_data_from_numpy(attention_mask)

        outputs = self.triton_client.infer(
            model_name=self.model_name, inputs=[input_ids_tensor, attention_mask_tensor]
        ).as_numpy("sentence_embedding")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        return outputs

    def load(self):
        self.triton_client.load_model(self.model_name)

    def unload(self):
        self.triton_client.unload_model(self.model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_name)


class TritonSentenceTransformersModelClient(TritonModelLoadingClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)

        self.model_name = get_model_name(model)

        if not self.triton_client.is_model_ready(self.model_name):
            _, self.tokenizer = create_model(model, triton_model_repository_path)
            self.triton_client.load_model(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.modalities = {"text"}

    def load(self):
        self.triton_client.load_model(self.model_name)

    def unload(self):
        self.triton_client.unload_model(self.model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_name)
