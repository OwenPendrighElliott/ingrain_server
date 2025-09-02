import os
import json
from sentence_transformers import SentenceTransformer
from ingrain_common.common import (
    save_library_name,
    custom_model_exists,
    save_model_source_name,
)
from ingrain_models.models.triton_sentence_transformers.sentence_transformer_converting import (
    onnx_transformer_model,
    generate_text_sentence_transformer_config,
)


def create_model(
    model_name: str,
    triton_model_repository_path: str,
    custom_model_dir: str,
    friendly_name: str,
) -> None:

    if custom_model_exists(custom_model_dir, model_name):
        with open(
            os.path.join(custom_model_dir, model_name, "_ingrain_model_meta.json"), "r"
        ) as f:
            model_meta = json.load(f)
        if model_meta["model_type"] != "sentence_transformers":
            raise ValueError(
                f"The custom model {model_name} exists but it is not a sentence_transformers model."
            )

        model = SentenceTransformer(os.path.join(custom_model_dir, model_name))
    else:
        model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    os.makedirs(
        os.path.join(triton_model_repository_path, friendly_name, "1"), exist_ok=True
    )

    text_encoder_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_name, "1", "model.onnx")
    )
    text_config_exists = os.path.exists(
        os.path.join(triton_model_repository_path, friendly_name, "config.pbtxt")
    )

    # exit early if the models and configs already exist
    if text_encoder_exists and text_config_exists:
        return friendly_name, tokenizer

    onnx_transformer_model(
        model,
        os.path.join(triton_model_repository_path, friendly_name, "1", "model.onnx"),
    )

    generate_text_sentence_transformer_config(
        os.path.join(triton_model_repository_path, friendly_name),
        friendly_name,
        model.get_sentence_embedding_dimension(),
    )

    save_library_name(
        os.path.join(triton_model_repository_path, friendly_name),
        "sentence_transformers",
    )

    save_model_source_name(
        os.path.join(triton_model_repository_path, friendly_name), model_name
    )

    with open(
        os.path.join(
            triton_model_repository_path,
            friendly_name,
            "sentence_transformer_config.json",
        ),
        "w+",
    ) as f:
        data = {"max_length": model._modules["0"].max_seq_length}
        json.dump(data, f)
