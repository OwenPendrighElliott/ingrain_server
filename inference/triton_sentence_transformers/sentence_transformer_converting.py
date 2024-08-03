import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List


class SentenceTransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SentenceTransformerWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model({"input_ids": input_ids, "attention_mask": attention_mask})[
            "sentence_embedding"
        ]


def generate_text_sentence_transformer_config(
    cfg_path: str,
    name: str,
    embedding_dim: int,
) -> None:
    config = f"""name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "sentence_embedding"
    data_type: TYPE_FP32
    dims: [ {embedding_dim} ]
  }}
]
dynamic_batching {{}}"""
    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def onnx_transformer_model(
    model: SentenceTransformer, output_path: str
) -> torch.jit.ScriptModule:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = SentenceTransformerWrapper(model)
    wrapped_model.to(device)

    dummy_input = {
        "input_ids": torch.tensor(
            [[101, 2023, 2003, 1037, 1398, 102]], dtype=torch.int64, device=device
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.int64, device=device),
    }

    torch.onnx.export(
        wrapped_model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        },
    )
