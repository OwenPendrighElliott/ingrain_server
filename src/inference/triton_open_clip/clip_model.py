import os
import open_clip
from open_clip.transform import image_transform_v2
from open_clip.transform import PreprocessCfg
import json
from ..common import get_text_image_model_names, save_library_name, custom_model_exists
from ..model_client import TritonModelLoadingClient
from .clip_converting import (
    onnx_convert_open_clip_model,
    generate_image_clip_config,
    generate_text_clip_config,
    image_transform_dict_from_torch_transforms
)

from typing import Optional, Union, List

_PREPROCESSOR_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "preprocessors")
PREPROCESS_CONFIGS = {
    os.path.basename(fname).split(".")[0]: json.load(
        open(os.path.join(_PREPROCESSOR_CONFIG_DIR, fname), "r")
    )
    for fname in os.listdir(_PREPROCESSOR_CONFIG_DIR)
}


def create_transforms(
    model_name: str, pretrained: Union[str, None], custom_model_dir: str
):
    """Create the transforms.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None]): The pretrained checkpoint.

    Returns:
        Tuple: The text and image model names, preprocess function, and tokenizer.
    """
    preprocessor_config = PREPROCESS_CONFIGS[model_name]
    if custom_model_exists(custom_model_dir, pretrained):
        with open(
            os.path.join(custom_model_dir, pretrained, "_ingrain_model_meta.json"), "r"
        ) as f:
            model_meta = json.load(f)
            del model_meta["model_type"]
            preprocessor_config.update(model_meta)

    preprocess = image_transform_v2(
        cfg=PreprocessCfg(**preprocessor_config),
        is_train=False,
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    friendly_text_name, friendly_image_name = get_text_image_model_names(
        model_name, pretrained
    )
    return friendly_text_name, friendly_image_name, preprocess, tokenizer


def create_model_and_transforms_triton(
    model_name: str,
    pretrained: Union[str, None],
    triton_model_repository_path: str,
    custom_model_dir: str,
):
    """Create the model and transforms for Triton.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None]): The pretrained checkpoint.
        triton_model_repository_path (str): The path to the Triton model repository.

    Returns:
        Tuple: The text and image model names, preprocess function, and tokenizer.
    """
    config = open_clip.get_model_config(model_name)
    if custom_model_exists(custom_model_dir, pretrained):
        model_file_path = os.path.join(
            custom_model_dir, pretrained, "model.safetensors"
        )
        model_config_path = os.path.join(
            custom_model_dir, pretrained, "_ingrain_model_meta.json"
        )

        with open(model_config_path, "r") as f:
            model_meta: dict = json.load(f)

        if not model_meta["model_type"] == "open_clip":
            raise ValueError(
                f"The custom model {model_name} exists but it is not an open_clip model."
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=model_file_path,
            image_mean=model_meta.get("mean"),
            image_std=model_meta.get("std"),
            image_interpolation=model_meta.get("interpolation"),
            image_resize_mode=model_meta.get("resize_mode"),
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    friendly_text_name, friendly_image_name = get_text_image_model_names(
        model_name, pretrained
    )
    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_text_name, "1"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_image_name, "1"),
        exist_ok=True,
    )

    image_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_image_name, "1", "model.pt")
    )
    text_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_text_name, "1", "model.pt")
    )
    image_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_image_name, "config.pbtxt")
    )
    text_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_text_name, "config.pbtxt")
    )

    # exit early if the models and configs already exist
    if (
        image_encoder_exists
        and text_encoder_exists
        and image_config_exists
        and text_config_exists
    ):
        return friendly_text_name, friendly_image_name, preprocess, tokenizer

    text_encoder_path = os.path.join(
        triton_model_repository_path, friendly_text_name, "1", "model.onnx"
    )

    image_encoder_path = os.path.join(
        triton_model_repository_path, friendly_image_name, "1", "model.onnx"
    )

    onnx_convert_open_clip_model(
        model,
        tokenizer,
        preprocess,
        text_encoder_path,
        image_encoder_path,
    )

    generate_text_clip_config(
        os.path.join(triton_model_repository_path, friendly_text_name),
        friendly_text_name,
        tokenizer.context_length,
        config["embed_dim"],
    )

    generate_image_clip_config(
        os.path.join(triton_model_repository_path, friendly_image_name),
        friendly_image_name,
        (3, *model.visual.preprocess_cfg["size"]),
        config["embed_dim"],
    )

    save_library_name(
        os.path.join(triton_model_repository_path, friendly_text_name), "open_clip"
    )
    save_library_name(
        os.path.join(triton_model_repository_path, friendly_image_name), "open_clip"
    )

    image_transform_config = image_transform_dict_from_torch_transforms(preprocess)
    transform_config_path = os.path.join(
        triton_model_repository_path, friendly_image_name, "image_transform_config.json"
    )
    with open(transform_config_path, "w") as f:
        json.dump(image_transform_config, f)

    return friendly_text_name, friendly_image_name

class TritonCLIPModelClient(TritonModelLoadingClient):
    def __init__(
        self,
        triton_grpc_url: str,
        model: str,
        pretrained: Union[str, None],
        triton_model_repository_path: str,
        custom_model_dir: str,
    ):
        super().__init__(triton_grpc_url)
        self.text_model_name, self.image_model_name = get_text_image_model_names(
            model, pretrained
        )

        if not self.triton_client.is_model_ready(
            self.text_model_name
        ) or not self.triton_client.is_model_ready(self.image_model_name):
            self.text_model_name, self.image_model_name = create_model_and_transforms_triton(
                model, pretrained, triton_model_repository_path, custom_model_dir
            )
            self.triton_client.load_model(self.text_model_name)
            self.triton_client.load_model(self.image_model_name)
        else:
            self.text_model_name, self.image_model_name = create_transforms(model, pretrained, custom_model_dir)

        self.modalities = {"text", "image"}

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
