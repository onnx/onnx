# SPDX-License-Identifier: Apache-2.0
"""function inline
"""

import onnx
from onnx import(NodeProto, ModelProto)
import onnx.onnx_cpp2py_export.function_inline as F
import sys

# Limitation of single protobuf file is 2GB
MAXIMUM_PROTOBUF = 2000000000

def inline_node_function(model: ModelProto) -> None:
    raise NotImplementedError

def inline_graph_function(model: ModelProto) -> None:
    raise NotImplementedError

def inline_model_function(source_model_path: str, target_model_path: str) -> None:
    """
    Inline functions in a model.
    """
    F.inline_model_function_path(source_model_path, target_model_path)

def inline_model_function(model: ModelProto) -> None:
    """Inline functions in a model.

    Arguments:
        model (ModelProto): model to inline
    """
    protobuf_string = model if isinstance(model, bytes) else model.SerializeToString()
    # If the protobuf is larger than 2GB,
    # remind users should use the model path to do funtion inline
    if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
        raise ValueError('This protobuf of onnx model is too large (>2GB). Call inline_model_function_path with model path instead.')
    inlined_model_str = F.inline_model_function(protobuf_string)
    return onnx.load_from_string(inlined_model_str)
