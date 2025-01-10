from typing import List
from PIL import Image
import numpy as np
import timm
import json
import os
import tritonclient.grpc as grpcclient
from .timm_converting import onnx_convert_timm_model, generate_timm_config
from ..model_client import TritonModelInferenceClient, TritonModelLoadingClient
from ..common import get_model_name, save_library_name, custom_model_exists


def create_model(
    model_name: str,
    pretrained: str | bool,
    triton_model_repository_path: str,
    custom_model_dir: str,
):
    friendly_name = get_model_name(model_name, pretrained)

    if isinstance(pretrained, str) and custom_model_exists(
        custom_model_dir, pretrained
    ):
        model_file_path = os.path.join(
            custom_model_dir, pretrained, "model.safetensors"
        )
        model_config_path = os.path.join(
            custom_model_dir, pretrained, "_ingrain_model_meta.json"
        )

        with open(model_config_path, "r") as f:
            model_meta = json.load(f)

        if not model_meta["model_type"] == "timm":
            raise ValueError(
                f"The custom model {model_name} exists but it is not a timm model."
            )

        model = timm.create_model(
            model_name,
            checkpoint_path=model_file_path,
            num_classes=model_meta["num_classes"],
        )
    elif isinstance(pretrained, bool):
        model = timm.create_model(model_name, pretrained=pretrained)
    else:
        raise ValueError(
            "Invalid pretrained value. It is string and is not a valid custom model, must be bool for timm models that are not custom."
        )

    model_cfg = timm.get_pretrained_cfg(model_name.split("/")[-1].split(".")[0])
    if model_cfg is None:
        model_cfg = timm.models.PretrainedCfg(
            input_size=model.pretrained_cfg["input_size"],
            num_classes=model.pretrained_cfg["num_classes"],
        )

    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_name, "1"), exist_ok=True
    )

    image_encoder_path = os.path.join(
        triton_model_repository_path, friendly_name, "1", "model.onnx"
    )

    if not os.path.exists(image_encoder_path):

        onnx_convert_timm_model(model, preprocess, image_encoder_path)
        cfg_path = os.path.join(triton_model_repository_path, friendly_name, "1")
        generate_timm_config(
            cfg_path, friendly_name, model_cfg.input_size, model_cfg.num_classes
        )

        save_library_name(
            os.path.join(triton_model_repository_path, friendly_name), "timm"
        )

        with open(
            os.path.join(
                triton_model_repository_path, friendly_name, "data_config.json"
            ),
            "w",
        ) as f:
            f.write(json.dumps(data_config))

    return friendly_name, preprocess


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
            with open(
                os.path.join(
                    triton_model_repository_path,
                    self.model_nice_name,
                    "data_config.json",
                ),
                "r",
            ) as f:
                data_config = json.load(f)
            self.preprocess = timm.data.create_transform(
                **data_config, is_training=False
            )

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

        processed_images = np.stack([self.preprocess(image).numpy() for image in image])

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


class TritonTimmModelClient(TritonModelLoadingClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: str | bool | None,
        triton_model_repository_path: str,
        custom_model_dir: str,
    ):
        super().__init__(triton_grpc_url)
        if pretrained is None:
            pretrained = True
        self.model_name = model
        self.model_nice_name = get_model_name(model, pretrained)
        self.triton_model_repository_path = triton_model_repository_path

        if not self.triton_client.is_model_ready(self.model_nice_name):
            _, _ = create_model(
                model, pretrained, triton_model_repository_path, custom_model_dir
            )
            self.triton_client.load_model(self.model_nice_name)

    def load(self):
        self.triton_client.load_model(self.model_nice_name)

    def unload(self):
        self.triton_client.unload_model(self.model_nice_name)

    def is_ready(self) -> bool:
        return self.triton_client.is_model_ready(self.model_nice_name)
