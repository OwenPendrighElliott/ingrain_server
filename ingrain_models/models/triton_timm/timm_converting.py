import os
import torch
from PIL import Image
from io import BytesIO
from torchvision.transforms import Compose, ToTensor, Resize
from timm.data import MaybeToTensor, MaybePILToTensor
import base64
from ingrain_models.models.triton_timm.timm_wrappers import TimmClassifierWrapper
from ingrain_common.common import (
    MAX_BATCH_SIZE,
    DYNAMIC_BATCHING,
    INSTANCE_KIND,
    MODEL_INSTANCES,
)
from typing import Tuple, Any


def convert_timm_to_onnx(
    model: torch.nn.Module, image: Image.Image, preprocess: Compose, output_path: str
) -> None:
    to_tensor_index = next(
        i
        for i, t in enumerate(preprocess.transforms)
        if isinstance(t, (ToTensor, MaybeToTensor, MaybePILToTensor))
    )
    model_with_baked_preprocess = TimmClassifierWrapper(model, preprocess)

    pre_tensor_transforms = Compose(
        transforms=preprocess.transforms[: to_tensor_index + 1]
    )

    image_dummy_input = pre_tensor_transforms(image).unsqueeze(0)

    torch.onnx.export(
        model_with_baked_preprocess,
        image_dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def generate_timm_config(
    cfg_path: str,
    name: str,
    image_shape: Tuple[int, int, int],
    num_classes: int,
) -> None:
    config = f"""
name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ {image_shape[0]}, {image_shape[1]}, {image_shape[2]} ]
  }}
]
output [
    {{
        name: "output"
        data_type: TYPE_FP32
        dims: [ {num_classes} ]
    }}
]
"""
    if DYNAMIC_BATCHING:
        config += f"\n\ndynamic_batching {{}}"

    if MODEL_INSTANCES > 0 and INSTANCE_KIND:
        f"""\n\ninstance_group [
            {{
                count: {MODEL_INSTANCES}
                kind: {INSTANCE_KIND}
            }}
        ]
        """

    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def onnx_convert_timm_model(
    model: torch.nn.Module,
    preprocess: Any,
    image_output_path: str,
) -> None:

    model.eval()
    image_base_64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA8UlEQVR4nOzQsQnAIAAAwRCySvafzdLOFfxKhLsJnv/+MR/2vKcDbmJWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFawAgAA///byQLaGwcUAQAAAABJRU5ErkJggg=="

    image = Image.open(BytesIO(base64.b64decode(image_base_64)))

    convert_timm_to_onnx(model, image, preprocess, image_output_path)
