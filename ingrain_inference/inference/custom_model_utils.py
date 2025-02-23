import os
import requests
import json
from .common import is_valid_dir_name

from typing import Optional, List


def download_custom_model(url: str, checkpoint_name: str, custom_model_dir: str) -> str:
    """Download a safetensors model file from a URL.

    Args:
        url (str): The URL to download the model from.
        checkpoint_name (str): The name of the checkpoint.
        custom_model_dir (str): The directory to save the model file.

    Returns:
        str: The path to the downloaded model file.
    """
    os.makedirs(custom_model_dir, exist_ok=True)
    os.makedirs(os.path.join(custom_model_dir, checkpoint_name), exist_ok=True)

    model_file_path = os.path.join(
        custom_model_dir, checkpoint_name, "model.safetensors"
    )

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(model_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=2**14):
                f.write(chunk)

    return model_file_path


def download_custom_model_auxillary_file(
    url: str, checkpoint_name: str, custom_model_dir: str, fname: str
) -> str:
    """Download auxillary files like tokenizer files for a custom model.

    Args:
        url (str): The URL to download the model from.
        checkpoint_name (str): The name of the checkpoint.
        custom_model_dir (str): The directory to save the model file.
        ext (str): The extension of the file.

    Returns:
        str: The path to the downloaded model file.
    """

    os.makedirs(custom_model_dir, exist_ok=True)

    file_path = os.path.join(custom_model_dir, checkpoint_name, fname)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=2**10):
                f.write(chunk)

    return file_path


def check_sentence_transformer_params(
    config_json_url: str,
    tokenizer_json_url: str,
    tokenizer_config_json_url: str,
    vocab_txt_url: str,
    special_tokens_map_json_url: str,
    pooling_config_json_url: str,
    sentence_bert_config_json_url: str,
    modules_json_url: str,
) -> None:
    """Check if the sentence transformer parameters are valid.

    Args:
        config_json_url (str): The URL to download the config.json file from.
        tokenizer_json_url (str): The URL to download the tokenizer.json file from.
        tokenizer_config_json_url (str): The URL to download the tokenizer_config.json file from.
        vocab_txt_url (str): The URL to download the vocab.txt file from.
        special_tokens_map_json_url (str): The URL to download the special_tokens_map.json file from.
        pooling_config_json_url (str): The URL to download the 1_Pooling/config.json file from.
        sentence_bert_config_json_url (str): The URL to download the sentence_bert_config.json file from.
        modules_json_url (str): The URL to download the modules.json file from.
    """
    missing_params = []
    if not config_json_url:
        missing_params.append("config_json_url")
    if not tokenizer_json_url:
        missing_params.append("tokenizer_json_url")
    if not tokenizer_config_json_url:
        missing_params.append("tokenizer_config_json_url")
    if not vocab_txt_url:
        missing_params.append("vocab_txt_url")
    if not special_tokens_map_json_url:
        missing_params.append("special_tokens_map_json_url")
    if not pooling_config_json_url:
        missing_params.append("pooling_config_json_url")
    if not sentence_bert_config_json_url:
        missing_params.append("sentence_bert_config_json_url")
    if not modules_json_url:
        missing_params.append("modules_json_url")

    if missing_params:
        raise ValueError(
            f"Missing parameters: {', '.join(missing_params)} for sentence transformer model, these auxillary parameters are required to load the model."
        )


def download_custom_sentence_transformers_model(
    custom_model_dir: str,
    model_name: str,
    model_url: str,
    config_json_url: str,
    tokenizer_json_url: str,
    tokenizer_config_json_url: str,
    vocab_txt_url: str,
    special_tokens_map_json_url: str,
    pooling_config_json_url: str,
    sentence_bert_config_json_url: str,
    modules_json_url: str,
) -> None:
    """Download a sentence transformers model from a URL.

    Args:
        custom_model_dir (str): The directory to save the model file.
        model_name (str): The name of the model.
        model_url (str): The URL to download the model from.
        config_json_url (str): The URL to download the config.json file from.
        tokenizer_json_url (str): The URL to download the tokenizer.json file from.
        tokenizer_config_json_url (str): The URL to download the tokenizer_config.json file from.
        vocab_txt_url (str): The URL to download the vocab.txt file from.
        special_tokens_map_json_url (str): The URL to download the special_tokens_map.json file from.
        pooling_config_json_url (str): The URL to download the 1_Pooling/config.json file from.
        sentence_bert_config_json_url (str): The URL to download the sentence_bert_config.json file from.
        modules_json_url (str): The URL to download the modules.json file from.
    """

    if not is_valid_dir_name(model_name):
        raise ValueError(
            f"Model name {model_name} is not a valid directory name, please choose a different name"
        )

    check_sentence_transformer_params(
        config_json_url,
        tokenizer_json_url,
        tokenizer_config_json_url,
        vocab_txt_url,
        special_tokens_map_json_url,
        pooling_config_json_url,
        sentence_bert_config_json_url,
        modules_json_url,
    )

    if os.path.exists(os.path.join(custom_model_dir, model_name)):
        raise ValueError(f"Model {model_name} already exists in {custom_model_dir}")

    download_custom_model(model_url, model_name, custom_model_dir)
    download_custom_model_auxillary_file(
        config_json_url, model_name, custom_model_dir, "config.json"
    )
    download_custom_model_auxillary_file(
        tokenizer_json_url, model_name, custom_model_dir, "tokenizer.json"
    )
    download_custom_model_auxillary_file(
        tokenizer_config_json_url, model_name, custom_model_dir, "tokenizer_config.json"
    )
    download_custom_model_auxillary_file(
        vocab_txt_url, model_name, custom_model_dir, "vocab.txt"
    )
    download_custom_model_auxillary_file(
        special_tokens_map_json_url,
        model_name,
        custom_model_dir,
        "special_tokens_map.json",
    )

    download_custom_model_auxillary_file(
        modules_json_url,
        model_name,
        custom_model_dir,
        "modules.json",
    )

    pooling_dir = os.path.join(custom_model_dir, model_name, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)
    download_custom_model_auxillary_file(
        pooling_config_json_url,
        model_name,
        custom_model_dir,
        os.path.join("1_Pooling", "config.json"),
    )

    download_custom_model_auxillary_file(
        sentence_bert_config_json_url,
        model_name,
        custom_model_dir,
        "sentence_bert_config.json",
    )

    with open(
        os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "w"
    ) as f:
        f.write(json.dumps({"model_type": "sentence_transformers"}))


def download_custom_open_clip_model(
    custom_model_dir: str,
    model_name: str,
    model_url: str,
    mode: Optional[str] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    interpolation: Optional[str] = None,
    resize_mode: Optional[str] = None,
) -> None:
    """Download a OpenCLIP model from a URL.

    Args:
        custom_model_dir (str): The directory to save the model file.
        model_name (str): The name of the model.
        model_url (str): The URL to download the model from.
    """

    if not is_valid_dir_name(model_name):
        raise ValueError(
            f"Model name {model_name} is not a valid directory name, please choose a different name"
        )

    if os.path.exists(os.path.join(custom_model_dir, model_name)):
        raise ValueError(f"Model {model_name} already exists in {custom_model_dir}")

    download_custom_model(model_url, model_name, custom_model_dir)

    model_data = {
        "model_type": "open_clip",
    }

    if mode:
        model_data["mode"] = mode
    if mean:
        model_data["mean"] = mean
    if std:
        model_data["std"] = std
    if interpolation:
        model_data["interpolation"] = interpolation
    if resize_mode:
        model_data["resize_mode"] = resize_mode

    with open(
        os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "w"
    ) as f:
        f.write(json.dumps(model_data))


def check_timm_params(
    num_classes: int,
) -> None:
    """Check if the timm model parameters are valid.

    Args:
        num_classes (int): The number of classes in the model.
    """
    if num_classes is None:
        raise ValueError("num_classes is required for timm model")


def download_custom_timm_model(
    custom_model_dir: str,
    model_name: str,
    model_url: str,
    num_classes: int,
) -> None:
    """Download a timm model from a URL.

    Args:
        custom_model_dir (str): The directory to save the model file.
        model_name (str): The name of the model.
        model_url (str): The URL to download the model from.
        num_classes (int): The number of classes in the model.
    """

    if not is_valid_dir_name(model_name):
        raise ValueError(
            f"Model name {model_name} is not a valid directory name, please choose a different name"
        )

    check_timm_params(num_classes)

    if os.path.exists(os.path.join(custom_model_dir, model_name)):
        raise ValueError(f"Model {model_name} already exists in {custom_model_dir}")

    download_custom_model(model_url, model_name, custom_model_dir)

    with open(
        os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "w"
    ) as f:
        f.write(json.dumps({"model_type": "timm", "num_classes": num_classes}))
