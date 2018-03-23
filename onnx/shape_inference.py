"""onnx shape inference. Shape inference is not guaranteed to be
complete.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.onnx_cpp2py_export.shape_inference as C

"""Apply shape inference to the provided ModelProto.

Inferred shapes are added to the value_info field of the graph.

If the inferred values conflict with values already provided in the
graph, that means that the provided values are invalid (or there is a
bug in shape inference), and the result is unspecified.

Arguments:
    input (string): serialized ModelProto
"""
infer_shapes = C.infer_shapes
