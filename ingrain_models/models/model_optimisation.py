import onnx
from onnxconverter_common import float16
from ingrain_common.common import FP16_ENABLED, MAX_BATCH_SIZE

from typing import List, Dict


def convert_to_float16(onnx_model_path: str, output_path: str) -> None:
    """
    Convert an ONNX model to float16 precision.

    Args:
        onnx_model_path (str): Path to the input ONNX model.
        output_path (str): Path to save the converted float16 ONNX model.
    """
    model = onnx.load(onnx_model_path)
    model_float16 = float16.convert_float_to_float16(model)
    onnx.save(model_float16, output_path)


def generate_tensorrt_config(input_shapes: Dict[str, List[int]]) -> str:
    if FP16_ENABLED:
        precision_mode = "FP16"
    else:
        precision_mode = "FP32"

    warmup_batches = ""
    for i in range(MAX_BATCH_SIZE):
        batch_size = i + 1

        inputs = ""
        for shape_name in input_shapes:
            inputs += f"""        inputs {{
            key: "{shape_name}"
            value: {{
                data_type: TYPE_FP32
                dims: [ {', '.join(map(str, input_shapes[shape_name]))} ]
                random_data: true
            }}
        }}
        """

        warmup_batches += f"""{{
    name : "batch {batch_size}"
    batch_size: {batch_size}
    {inputs}
    count: 1
  }}"""

    warmup_config = f"""
model_warmup [
  {warmup_batches}
]
"""

    return f"""
optimization {{ execution_accelerators {{
  gpu_execution_accelerator : [ {{
    name : "tensorrt"
    parameters {{ key: "precision_mode" value: "{precision_mode}" }}
    parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
    }}]
}}}}

{warmup_config}
"""
