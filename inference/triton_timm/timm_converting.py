import os
import torch
from PIL import Image
from io import BytesIO
import base64
from open_clip import CustomTextCLIP, CLIP
from typing import Tuple, Any


def convert_timm_to_onnx(
    model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str
) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        do_constant_folding=True,
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
max_batch_size: 32
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
dynamic_batching {{}}"""
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

    image_dummy_input = preprocess(image).unsqueeze(0)
    convert_timm_to_onnx(model, image_dummy_input, image_output_path)
