import os
import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from io import BytesIO
import base64
from open_clip import CustomTextCLIP, CLIP
from typing import Tuple, Any
from ingrain_models.models.triton_open_clip.open_clip_wrappers import (
    CLIPTextEncoderWrapper,
    CLIPImageEncoderWrapper,
)
from ingrain_models.models.model_optimisation import (
    generate_tensorrt_config,
    optimize_onnx_model,
)
from ingrain_common.common import (
    MAX_BATCH_SIZE,
    DYNAMIC_BATCHING,
    MODEL_INSTANCES,
    INSTANCE_KIND,
    TENSORRT_ENABLED,
)


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

    to_tensor_index = next(
        i for i, t in enumerate(preprocess.transforms) if isinstance(t, ToTensor)
    )
    model_with_baked_preprocess = CLIPImageEncoderWrapper(model, preprocess)

    pre_tensor_transforms = Compose(
        transforms=preprocess.transforms[: to_tensor_index + 1]
    )

    image_dummy_input = pre_tensor_transforms(dummy_input).unsqueeze(0)

    torch.onnx.export(
        model_with_baked_preprocess,
        image_dummy_input,
        output_path,
        export_params=True,
        opset_version=20,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    optimize_onnx_model(output_path, output_path)


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
        opset_version=20,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    optimize_onnx_model(output_path, output_path)


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
"""
    if DYNAMIC_BATCHING:
        config += "\n\ndynamic_batching {}"

    if MODEL_INSTANCES > 0 and INSTANCE_KIND:
        config += f"""\n\ninstance_group [
    {{
        count: {MODEL_INSTANCES}
        kind: {INSTANCE_KIND}
    }}
]
        """

    if TENSORRT_ENABLED:
        tensorrt_config = generate_tensorrt_config({"input": [context_length]}, "INT32")
        config += f"\n\n{tensorrt_config}"

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
"""
    if DYNAMIC_BATCHING:
        config += "\n\ndynamic_batching {}"

    if MODEL_INSTANCES > 0 and INSTANCE_KIND:
        config += f"""\n\ninstance_group [
    {{
        count: {MODEL_INSTANCES}
        kind: {INSTANCE_KIND}
    }}
]
        """

    if TENSORRT_ENABLED:
        tensorrt_config = generate_tensorrt_config({"input": image_shape}, "FP32")
        config += f"\n\n{tensorrt_config}"

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
