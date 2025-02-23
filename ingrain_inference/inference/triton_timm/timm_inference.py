import json
import os
from PIL import Image
import numpy as np
from ..preprocessors.image_preprocessor import load_image_transform_config
import tritonclient.grpc as grpcclient
from ..model_client import TritonModelInferenceClient
from ..common import get_model_name

from typing import List


class TritonTimmInferenceClient(TritonModelInferenceClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: str | bool | None,
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)
        self.model_name = model

        if pretrained is None:
            pretrained = True
        self.model_nice_name = get_model_name(model, pretrained)

        if not self.triton_client.is_model_ready(self.model_nice_name):
            raise ValueError(f"Model {self.model_name} is not ready on the server.")
        else:
            preprocess_config_path = os.path.join(
                triton_model_repository_path,
                self.model_nice_name,
                "image_transform_config.json",
            )
            self.preprocess = load_image_transform_config(preprocess_config_path)

        self.modalities = {"image"}

    def encode_image(
        self,
        image: Image.Image | List[Image.Image],
        normalize: bool = True,
        n_dims: int | None = None,
    ) -> np.ndarray:
        if n_dims is not None:
            raise ValueError("Timm models do not support n_dims parameter.")

        if isinstance(image, Image.Image):
            image = [image]

        processed_images = np.stack([self.preprocess(image) for image in image])

        image_inputs = grpcclient.InferInput("input", processed_images.shape, "FP32")
        image_inputs.set_data_from_numpy(processed_images)
        outputs = self.triton_client.infer(
            model_name=self.model_nice_name, inputs=[image_inputs]
        ).as_numpy("output")

        if normalize:
            # convert to logits
            outputs = np.exp(outputs) / np.exp(outputs).sum(axis=-1, keepdims=True)

        return outputs

    def load(self):
        self.triton_client.load_model(self.model_nice_name)

    def unload(self):
        self.triton_client.unload_model(self.model_nice_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_nice_name)
