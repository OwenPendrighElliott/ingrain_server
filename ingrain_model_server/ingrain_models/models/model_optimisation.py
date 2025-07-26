import onnx
from onnxconverter_common import float16
from onnxoptimizer import optimize
from ingrain_common.common import FP16_ENABLED, MAX_BATCH_SIZE

from typing import List, Dict, Literal


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


def optimize_onnx_model(onnx_model_path: str, output_path: str) -> None:
    """
    Optimize an ONNX model using onnxoptimizer.

    Args:
        onnx_model_path (str): Path to the input ONNX model.
        output_path (str): Path to save the optimized ONNX model.
    """
    model = onnx.load(onnx_model_path)
    optimized_model = optimize(
        model,
        passes=[
            "eliminate_nop_dropout",
            "eliminate_nop_flatten",
            "eliminate_if_with_const_cond",
            "eliminate_nop_concat",
            "eliminate_nop_split",
            "eliminate_nop_expand",
            "eliminate_shape_gather",
            "eliminate_slice_after_shape",
            "eliminate_nop_transpose",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_log_softmax",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_pad_into_pool",
            "fuse_transpose_into_gemm",
            "replace_einsum_with_matmul",
            "fuse_concat_into_reshape",
            "eliminate_nop_reshape",
            "eliminate_nop_with_unit",
            "fuse_qkv",
            "fuse_consecutive_unsqueezes",
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_shape_op",
            "fuse_consecutive_slices",
            "adjust_slice_and_matmul",
        ],
    )
    onnx.save(optimized_model, output_path)


def generate_tensorrt_config(
    input_shapes: Dict[str, List[int]],
    input_dtype: Literal["FP16", "FP32", "INT32", "INT64"],
) -> str:
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
                data_type: TYPE_{input_dtype}
                dims: [ {', '.join(map(str, input_shapes[shape_name]))} ]
                random_data: true
            }}
        }},
        """

        warmup_batches += f"""{{
    name : "batch {batch_size}"
    batch_size: {batch_size}
    {inputs}
    count: 1
  }},"""

    warmup_config = f"""
model_warmup [
  {warmup_batches[:-1]}
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
