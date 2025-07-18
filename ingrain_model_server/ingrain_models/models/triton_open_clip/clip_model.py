import os
import open_clip
from open_clip.transform import image_transform_v2
from open_clip.transform import PreprocessCfg
from huggingface_hub import hf_hub_download
import json
from ingrain_common.common import (
    save_library_name,
    custom_model_exists,
)
from ingrain_models.models.triton_open_clip.clip_converting import (
    onnx_convert_open_clip_model,
    generate_image_clip_config,
    generate_text_clip_config,
)
from ingrain_models.models.triton_open_clip.tokenizer_tools.export_tokenizer import (
    export_tokenizer,
)

from ingrain_models.models.torchvision_transform_conversion import (
    image_transform_dict_from_torch_transforms,
)

from typing import Union


def create_model_and_transforms_triton(
    model_name: str,
    pretrained: Union[str, None],
    triton_model_repository_path: str,
    custom_model_dir: str,
    friendly_text_name: str,
    friendly_image_name: str,
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
    if config is None and model_name.startswith("hf-hub:"):
        config_path = hf_hub_download(
            repo_id=model_name.split("hf-hub:")[1],
            library_name="open_clip",
            filename="open_clip_config.json",
        )
        with open(config_path, "r") as f:
            config = json.load(f)["model_cfg"]

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

    image_size = model.visual.preprocess_cfg["size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    generate_image_clip_config(
        os.path.join(triton_model_repository_path, friendly_image_name),
        friendly_image_name,
        (3, *image_size),
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

    export_tokenizer(
        tokenizer,
        os.path.join(triton_model_repository_path, friendly_text_name, "tokenizer"),
    )

    return friendly_text_name, friendly_image_name
