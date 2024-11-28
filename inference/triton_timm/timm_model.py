from typing import List
from PIL import Image
import numpy as np
import timm
import os
import tritonclient.grpc as grpcclient
from .timm_converting import onnx_convert_timm_model, generate_timm_config
from ..model_client import TritonModelInferenceClient, TritonModelLoadingClient
from ..common import get_model_name

class TritonTimmInferenceClient(TritonModelInferenceClient):
    def __init__(
        self, 
        triton_grpc_url: str,
        model: str,
    ):
        super().__init__(triton_grpc_url)
        self.model_name = model
        self.model_nice_name = get_model_name(model)

        if not self.triton_client.is_model_ready(self.model_nice_name):
            raise ValueError(f"Model {self.model_name} is not ready on the server.")
        else:
            config = timm.get_pretrained_cfg(self.model_name.split("/")[-1].split(".")[0])
            config_dict = vars(config)
            self.preprocess = timm.data.create_transform(config_dict, is_training=False)

        self.modalities = {"image"}

    def encode_image(self, image: Image.Image | List[Image.Image], normalize: bool = True, n_dims: int | None = None) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = [image]
        
        processed_images = np.stack([self.preprocess(image).numpy() for image in image])

        image_inputs = grpcclient.InferInput("input", processed_images.shape, "FP32")
        image_inputs.set_data_from_numpy(processed_images)
        outputs = self.triton_client.infer(
            model_name=self.model_nice_name, inputs=[image_inputs]
        ).as_numpy("output")

        if normalize:
            outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)

        if n_dims is not None:
            outputs = outputs[:, :n_dims]

        return outputs

    def load(self):
        self.triton_client.load_model(self.model_nice_name)

    def unload(self):
        self.triton_client.unload_model(self.model_nice_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_nice_name)


class TritonTimmModelClient(TritonModelLoadingClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        triton_model_repository_path: str,
    ):
        super().__init__(triton_grpc_url)
        self.model_name = model
        self.model_nice_name = get_model_name(model)
        self.triton_model_repository_path = triton_model_repository_path

        if not self.triton_client.is_model_ready(self.model_nice_name):
            raise ValueError(f"Model {self.model_name} is not ready on the server.")
        else:
            self.model = timm.create_model(self.model_name, pretrained=True)
            model_cfg = timm.get_pretrained_cfg(self.model_name.split("/")[-1].split(".")[0])
            data_config = timm.data.resolve_model_data_config(model)
            self.preprocess = timm.data.create_transform(**data_config, is_training=False)
            image_encoder_path = os.path.join(triton_model_repository_path, self.model_nice_name, "1", "model.onnx")
            onnx_convert_timm_model(self.model, self.preprocess, image_encoder_path)
            cfg_path = os.path.join(triton_model_repository_path, self.model_nice_name, "1")
            generate_timm_config(cfg_path, self.model_nice_name, model_cfg.input_size, model_cfg.num_classes)
    
    def load(self):
        self.triton_client.load_model(self.model_nice_name)

    def unload(self):
        self.triton_client.unload_model(self.model_nice_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_nice_name)