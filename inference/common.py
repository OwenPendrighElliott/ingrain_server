import os
import shutil
import numpy as np
from typing import Union, Tuple


def get_model_name(model_name: str, pretrained: Union[str, None] = None) -> str:
    """Get the model name.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None], optional): The pretrained checkpoint. Defaults to None.

    Returns:
        str: The model name.
    """
    name = model_name.replace("/", "_")
    if pretrained is not None:
        name += f"_{pretrained}"
    return name


def get_text_image_model_names(
    model_name: str, pretrained: Union[str, None] = None
) -> Tuple[str]:
    """Get the text and image encoder model names.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None], optional): The pretrained checkpoint. Defaults to None.

    Returns:
        Tuple[str]: The text and image encoder model names.
    """
    name = model_name.replace("/", "_")
    if pretrained is not None:
        name += f"_{pretrained}"

    text_encoder_name = name + "_text_encoder"
    image_encoder_name = name + "_image_encoder"
    return text_encoder_name, image_encoder_name


def delete_model_from_repo(
    model_name: str, pretrained: Union[str, None], triton_model_repository_path: str
) -> None:
    """Delete the model from the Triton model repository.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None]): The pretrained checkpoint.
        triton_model_repository_path (str): The path to the Triton model repository.
    """
    friendly_name = get_model_name(model_name, pretrained)
    model_path_sentence_transformer = os.path.join(
        triton_model_repository_path, friendly_name
    )
    if os.path.exists(model_path_sentence_transformer):
        shutil.rmtree(model_path_sentence_transformer)
        return

    text_encoder_name, image_encoder_name = get_text_image_model_names(
        model_name, pretrained
    )
    model_path_image_encoder = os.path.join(
        triton_model_repository_path, image_encoder_name
    )
    model_path_text_encoder = os.path.join(
        triton_model_repository_path, text_encoder_name
    )
    if os.path.exists(model_path_image_encoder) and os.path.exists(
        model_path_text_encoder
    ):
        shutil.rmtree(model_path_image_encoder)
        shutil.rmtree(model_path_text_encoder)
        return
