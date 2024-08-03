import os
import torch

from typing import Tuple, List


class CLIPImageEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super(CLIPImageEncoderWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_image(x)


class CLIPTextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super(CLIPTextEncoderWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode_text(x)


def generate_image_clip_config(
    cfg_path: str,
    name: str,
    image_shape: Tuple[int, int, int],
    embedding_dim: int,
    preferred_batch_sizes: List[int],
    max_queue_delay_microseconds: int,
) -> None:
    config = f"""
name: "{name}"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ {image_shape[0]}, {image_shape[1]}, {image_shape[2]} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {embedding_dim} ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ {", ".join([str(v) for v in preferred_batch_sizes])} ]
  max_queue_delay_microseconds: {max_queue_delay_microseconds}
}}"""
    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def generate_text_clip_config(
    cfg_path: str,
    name: str,
    context_length: int,
    embedding_dim: int,
    preferred_batch_sizes: List[int],
    max_queue_delay_microseconds: int,
) -> None:
    config = f"""name: "{name}"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {{
    name: "input__0"
    data_type: TYPE_INT64
    dims: [ {context_length} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {embedding_dim} ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ {", ".join([str(v) for v in preferred_batch_sizes])} ]
  max_queue_delay_microseconds: {max_queue_delay_microseconds}
}}"""
    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def script_open_clip_model(
    model: torch.nn.Module,
) -> Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    image_encoder = CLIPImageEncoderWrapper(model)
    text_encoder = CLIPTextEncoderWrapper(model)
    image_encoder.eval()
    text_encoder.eval()
    image_encoder = torch.jit.script(image_encoder)
    text_encoder = torch.jit.script(text_encoder)
    return image_encoder, text_encoder
