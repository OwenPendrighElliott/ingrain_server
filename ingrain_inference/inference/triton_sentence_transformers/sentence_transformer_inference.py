import os
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tokenizers import Tokenizer, Encoding
from ..model_client import TritonModelInferenceClient
from ..common import get_model_name, custom_model_exists

from typing import Union, List, Optional


def get_sentence_transformers_parameters(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "sentence_transformer_config.json"), "r") as f:
        params = json.load(f)
    return params


class TritonSentenceTransformersInferenceClient(TritonModelInferenceClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        custom_model_dir: str,
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)

        self.model_name = get_model_name(model)
        self.meta_parameters = {}
        if not self.triton_client.is_model_ready(self.model_name):
            raise ValueError(f"Model {model} is not ready")
        else:
            if custom_model_exists(custom_model_dir, model):
                with open(
                    os.path.join(custom_model_dir, model, "_ingrain_model_meta.json"),
                    "r",
                ) as f:
                    model_meta = json.load(f)
                if model_meta["model_type"] != "sentence_transformers":
                    raise ValueError(
                        f"The custom model {model} exists but it is not a sentence_transformers model."
                    )
                self.meta_parameters = get_sentence_transformers_parameters(
                    os.path.join(custom_model_dir, model)
                )
                self.tokenizer = Tokenizer.from_pretrained(
                    os.path.join(custom_model_dir, model)
                )
                self.tokenizer.enable_truncation(
                    max_length=self.meta_parameters["max_length"]
                )
            else:
                self.meta_parameters = get_sentence_transformers_parameters(
                    os.path.join(triton_model_repository_path, self.model_name)
                )
                self.tokenizer: Tokenizer = Tokenizer.from_pretrained(model)
                self.tokenizer.enable_truncation(
                    max_length=self.meta_parameters["max_length"]
                )

        self.modalities = {"text"}

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        n_dims: Optional[int] = None,
    ) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        encoding: List[Encoding] = self.tokenizer.encode_batch(
            text, is_pretokenized=False
        )
        max_len = max([len(e.ids) for e in encoding])
        input_ids = np.zeros((len(encoding), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encoding), max_len), dtype=np.int64)

        for i, e in enumerate(encoding):
            input_ids[i, : len(e.ids)] = e.ids
            attention_mask[i, : len(e.ids)] = 1

        # input_ids = tokens["input_ids"].astype(np.int64)
        # attention_mask = tokens["attention_mask"].astype(np.int64)

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

        if n_dims is not None:
            outputs = outputs[:, :n_dims]

        return outputs

    def load(self):
        self.triton_client.load_model(self.model_name)

    def unload(self):
        self.triton_client.unload_model(self.model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_name)
