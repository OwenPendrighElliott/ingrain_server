import timm
import json
import os
from ingrain_models.models.triton_timm.timm_converting import (
    onnx_convert_timm_model,
    generate_timm_config,
)
from ingrain_models.models.torchvision_transform_conversion import (
    image_transform_dict_from_torch_transforms,
)

from ingrain_common.common import save_library_name, custom_model_exists


def create_model(
    model_name: str,
    pretrained: str | None,
    triton_model_repository_path: str,
    custom_model_dir: str,
    friendly_name: str,
) -> None:

    if pretrained is not None and custom_model_exists(custom_model_dir, pretrained):
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
    model = timm.create_model(model_name, pretrained=True)

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

        image_transform_config = image_transform_dict_from_torch_transforms(preprocess)
        transform_config_path = os.path.join(
            triton_model_repository_path, friendly_name, "image_transform_config.json"
        )
        with open(transform_config_path, "w") as f:
            json.dump(image_transform_config, f)

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
