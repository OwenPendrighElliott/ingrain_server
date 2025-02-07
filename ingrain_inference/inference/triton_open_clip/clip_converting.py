import os
import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
from io import BytesIO
import base64
from open_clip import CustomTextCLIP, CLIP
from typing import Tuple, Any
from .open_clip_wrappers import CLIPTextEncoderWrapper, CLIPImageEncoderWrapper
from ..common import MAX_BATCH_SIZE

from typing import List


def image_transform_dict_from_torch_transforms(transforms: Compose) -> List[dict]:
    transform_dict = []
    for transform in transforms.transforms:
        if isinstance(transform, Resize):
            size = transform.size
            if isinstance(size, int):
                size = (size, size)
            transform_data = {
                "type": "ResizeImage",
                "size": size,
                "method": transform.interpolation,
            }
            transform_dict.append(transform_data)
        elif transform.__name__ == "_convert_to_rgb":
            transform_data = {"type": "ConvertToRGB"}
            transform_dict.append(transform_data)

    return transform_dict


def decompose_clip_preprocess(preprocess: Compose) -> Tuple[Compose, nn.Sequential]:
    to_tensor_index = preprocess.transforms.index(ToTensor)
    pre_tensor_transforms = Compose(
        transforms=preprocess.transforms[: to_tensor_index + 1]
    )
    post_tensor_transforms = nn.Sequential(preprocess.transforms[to_tensor_index + 1 :])
    return pre_tensor_transforms, post_tensor_transforms


def convert_image_encoder_to_onnx(
    model: torch.nn.Module,
    dummy_input: Image.Image,
    preprocess: Compose,
    output_path: str,
) -> None:

    to_tensor_index = preprocess.transforms.index(ToTensor)
    model_with_baked_preprocess = CLIPImageEncoderWrapper(model, preprocess)

    pre_tensor_transforms = Compose(
        transforms=preprocess.transforms[: to_tensor_index + 1]
    )

    image_dummy_input = pre_tensor_transforms(dummy_input)

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

    convert_text_encoder_to_onnx(model, text_dummy_input, text_output_path)
    convert_image_encoder_to_onnx(model, image, preprocess, image_output_path)
