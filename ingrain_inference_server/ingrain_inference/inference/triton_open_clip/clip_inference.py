import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tokenizers import Tokenizer, Encoding
import os
import json
from ingrain_inference.inference.preprocessors.image_preprocessor import (
    load_image_transform_config,
)
from ingrain_common.common import get_text_image_model_names
from ingrain_inference.inference.inference_client import TritonModelInferenceClient

from typing import Union, List, Optional


class TritonCLIPInferenceClient(TritonModelInferenceClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: Union[str, None],
        custom_model_dir: str,
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)
        self.text_model_name, self.image_model_name = get_text_image_model_names(
            model, pretrained
        )

        if not self.triton_client.is_model_ready(
            self.text_model_name
        ) or not self.triton_client.is_model_ready(self.image_model_name):
            raise ValueError(f"Model {model} is not ready")
        else:
            preprocess_config_path = os.path.join(
                triton_model_repository_path,
                self.image_model_name,
                "image_transform_config.json",
            )
            self.preprocess = load_image_transform_config(preprocess_config_path)
            self.tokenizer: Tokenizer = Tokenizer.from_file(
                os.path.join(
                    triton_model_repository_path,
                    self.text_model_name,
                    "tokenizer",
                    "tokenizer.json",
                )
            )

            with open(
                os.path.join(
                    triton_model_repository_path,
                    self.text_model_name,
                    "tokenizer",
                    "_tokenizer_context_length.json",
                ),
                "r",
            ) as f:
                self.context_length = json.load(f)["context_length"]

        self.modalities = {"text", "image"}

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

        for i in range(len(encoding)):
            if len(encoding[i].ids) > self.context_length:
                encoding[i].truncate(max_length=self.context_length)
            else:
                encoding[i].pad(length=self.context_length)

        tokens = np.stack([e.ids for e in encoding])
        tokens = tokens.astype(np.int32)

        text_inputs = grpcclient.InferInput("input", tokens.shape, "INT32")
        text_inputs.set_data_from_numpy(tokens)
        outputs = self.triton_client.infer(
            model_name=self.text_model_name, inputs=[text_inputs]
        ).as_numpy("output")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        if n_dims is not None:
            outputs = outputs[:, :n_dims]

        return outputs

    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        normalize: bool = True,
        n_dims: Optional[int] = None,
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = [image]
        processed_images = np.stack([self.preprocess(image) for image in image])

        image_inputs = grpcclient.InferInput("input", processed_images.shape, "FP32")
        image_inputs.set_data_from_numpy(processed_images)
        outputs = self.triton_client.infer(
            model_name=self.image_model_name, inputs=[image_inputs]
        ).as_numpy("output")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        if n_dims is not None:
            outputs = outputs[:, :n_dims]

        return outputs

    def load(self):
        self.triton_client.load_model(self.text_model_name)
        self.triton_client.load_model(self.image_model_name)

    def unload(self):
        self.triton_client.unload_model(self.text_model_name)
        self.triton_client.unload_model(self.image_model_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(
            self.text_model_name
        ) and self.triton_client.is_model_ready(self.image_model_name)
