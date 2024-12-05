import os
import requests
import json
from .common import is_valid_dir_name

from typing import Optional


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
    sentence_bert_config_json_url: Optional[str] = None,
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
    """

    if not is_valid_dir_name(model_name):
        raise ValueError(
            f"Model name {model_name} is not a valid directory name, please choose a different name"
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

    pooling_dir = os.path.join(custom_model_dir, model_name, "1_Pooling")
    os.makedirs(pooling_dir, exist_ok=True)
    download_custom_model_auxillary_file(
        pooling_config_json_url, model_name, pooling_dir, "config.json"
    )

    if sentence_bert_config_json_url:
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

    with open(
        os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "w"
    ) as f:
        f.write(json.dumps({"model_type": "open_clip"}))


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

    if os.path.exists(os.path.join(custom_model_dir, model_name)):
        raise ValueError(f"Model {model_name} already exists in {custom_model_dir}")

    download_custom_model(model_url, model_name, custom_model_dir)

    with open(
        os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "w"
    ) as f:
        f.write(json.dumps({"model_type": "timm", "num_classes": num_classes}))
