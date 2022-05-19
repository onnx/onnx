# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.onnx_cpp2py_export.printer as C
from typing import Text


def to_text(proto):  # type: (Text) -> onnx.ModelProto
    if isinstance(proto, onnx.FunctionProto):
        return C.function_to_text(proto.SerializeToString())
    elif isinstance(proto, onnx.GraphProto):
        return C.graph_to_text(proto.SerializeToString())
    return TypeError("Unsupported argument type.")