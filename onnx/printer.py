# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.onnx_cpp2py_export.printer as C
from typing import Text, Union


def to_text(proto: Union[onnx.ModelProto, onnx.FunctionProto, onnx.GraphProto]) -> Text:
    if isinstance(proto, onnx.ModelProto):
        return C.model_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.FunctionProto):
        return C.function_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.GraphProto):
        return C.graph_to_text(proto.SerializeToString())
    return TypeError("Unsupported argument type.")
