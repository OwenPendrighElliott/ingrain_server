import os
import shutil
import platform
from typing import Union, Tuple, Literal

MAX_BATCH_SIZE = os.getenv("MAX_BATCH_SIZE", 32)
DYNAMIC_BATCHING = os.getenv("DYNAMIC_BATCHING", "true").lower() == "true"
INSTANCE_KIND: Literal["KIND_GPU", "KIND_CPU", None] = os.getenv("INSTANCE_KIND")
TENSORRT_ENABLED = os.getenv("TENSORRT_ENABLED", "false").lower() == "true"
FP16_ENABLED = os.getenv("FP16_ENABLED", "false").lower() == "true"
MODEL_INSTANCES = int(os.getenv("MODEL_INSTANCES", -1))


def validate_env_vars() -> None:
    """Validate environment variables."""
    if not isinstance(MAX_BATCH_SIZE, int) or MAX_BATCH_SIZE <= 0:
        raise ValueError("MAX_BATCH_SIZE must be a positive integer.")

    if INSTANCE_KIND not in [None, "KIND_GPU", "KIND_CPU"]:
        raise ValueError(
            "INSTANCE_KIND must be either 'KIND_GPU', 'KIND_CPU', or None (not set)."
        )
    if INSTANCE_KIND and MODEL_INSTANCES == -1:
        raise ValueError(
            "MODEL_INSTANCES must be set to a non-negative integer when INSTANCE_KIND is set."
        )

    if INSTANCE_KIND and MODEL_INSTANCES < 0:
        raise ValueError(
            "MODEL_INSTANCES cannot be less that 0 when INSTANCE_KIND is set."
        )

    if INSTANCE_KIND != "KIND_GPU":
        if TENSORRT_ENABLED:
            raise ValueError(
                "TENSORRT_ENABLED can only be set to True when INSTANCE_KIND is 'KIND_GPU'."
            )
        if FP16_ENABLED:
            raise ValueError(
                "FP16_ENABLED can only be set to True when INSTANCE_KIND is 'KIND_GPU'."
            )


def get_model_name(model_name: str, pretrained: Union[str, bool, None] = None) -> str:
    """Get the model name.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None], optional): The pretrained checkpoint. Defaults to None.

    Returns:
        str: The model name.
    """
    name = model_name.replace("/", "_").replace(":", "___")
    if pretrained is not None and isinstance(pretrained, str):
        name += f"_{pretrained}"
    return name


def get_text_image_model_names(
    model_name: str, pretrained: Union[str, None] = None
) -> Tuple[str, str]:
    """Get the text and image encoder model names.

    Args:
        model_name (str): The model name.
        pretrained (Union[str, None], optional): The pretrained checkpoint. Defaults to None.

    Returns:
        Tuple[str]: The text and image encoder model names.
    """
    name = get_model_name(model_name, pretrained)

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


def save_library_name(output_dir: str, library_name: str):
    """Save the library name to a file.

    Args:
        output_dir (str): The output directory.
        library_name (str): The library name.
    """
    with open(os.path.join(output_dir, "library_name.txt"), "w") as f:
        f.write(library_name)


def get_library_name(output_dir: str) -> str:
    """Get the library name from a file.

    Args:
        output_dir (str): The output directory.

    Returns:
        str: The library name.
    """
    library_name_path = os.path.join(output_dir, "library_name.txt")
    if not os.path.exists(library_name_path):
        raise FileNotFoundError(f"Library name file not found in {output_dir}")

    with open(library_name_path, "r") as f:
        return f.read().strip()


def is_valid_dir_name(name: str) -> bool:
    invalid_chars = r'<>:"/\\|?*'
    reserved_names = (
        [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"{base}{num}" for base in ["COM", "LPT"] for num in range(1, 10)),
        ]
        if platform.system() == "Windows"
        else []
    )

    if not name or name.strip() == "":
        return False

    if any(char in name for char in invalid_chars):
        return False

    if platform.system() == "Windows" and name.upper() in reserved_names:
        return False

    if len(name) > 255:
        return False

    return True


def custom_model_exists(
    custom_model_dir: str,
    model_name: str | None,
) -> bool:
    """Check if the folder for the custom model exists.

    Args:
        custom_model_dir (str): The custom model directory.
        model_name (str): The model name.

    Returns:
        bool: Whether the custom model exists.
    """

    if model_name is None:
        return False

    if not os.path.exists(custom_model_dir):
        return False

    model_dir = os.path.join(custom_model_dir, model_name)
    if not os.path.exists(model_dir):
        return False

    return True
