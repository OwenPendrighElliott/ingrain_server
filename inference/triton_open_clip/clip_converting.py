import os
import torch
from PIL import Image
from io import BytesIO
import base64
from open_clip import CustomTextCLIP, CLIP
from typing import Tuple, Any
from .open_clip_wrappers import CLIPTextEncoderWrapper
from ..common import MAX_BATCH_SIZE


def convert_image_encoder_to_onnx(
    model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str
) -> None:
    torch.onnx.export(
        model.visual,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def convert_text_encoder_to_onnx(
    model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str
) -> None:
    if isinstance(model, CustomTextCLIP):
        text_tower = model.text
    elif isinstance(model, CLIP):
        text_tower = CLIPTextEncoderWrapper(model)
    dummy_input = dummy_input.to(torch.int32)
    torch.onnx.export(
        text_tower,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def generate_text_clip_config(
    cfg_path: str,
    name: str,
    context_length: int,
    embedding_dim: int,
) -> None:
    config = f"""
name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "input"
    data_type: TYPE_INT32
    dims: [ {context_length} ]
  }}
]
output [
    {{
        name: "output"
        data_type: TYPE_FP32
        dims: [ {embedding_dim} ]
    }}
]
dynamic_batching {{}}"""
    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def generate_image_clip_config(
    cfg_path: str,
    name: str,
    image_shape: Tuple[int, int, int],
    embedding_dim: int,
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
        dims: [ {embedding_dim} ]
    }}
]
dynamic_batching {{}}"""
    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def onnx_convert_open_clip_model(
    model: torch.nn.Module,
    tokenizer: Any,
    preprocess: Any,
    text_output_path: str,
    image_output_path: str,
) -> None:
    model.eval()

    text_dummy_input = tokenizer(["a photo of a dog", "a photo of a cat"])

    image_base_64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA8UlEQVR4nOzQsQnAIAAAwRCySvafzdLOFfxKhLsJnv/+MR/2vKcDbmJWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFawAgAA///byQLaGwcUAQAAAABJRU5ErkJggg=="

    image = Image.open(BytesIO(base64.b64decode(image_base_64)))

    image_dummy_input = preprocess(image).unsqueeze(0)

    convert_text_encoder_to_onnx(model, text_dummy_input, text_output_path)
    convert_image_encoder_to_onnx(model, image_dummy_input, image_output_path)
