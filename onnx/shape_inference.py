"""onnx shape inference. Shape inference is not guaranteed to be
complete.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.shape_inference as C
from onnx import ModelProto

"""Apply shape inference to the provided ModelProto.

Inferred shapes are added to the value_info field of the graph.

If the inferred values conflict with values already provided in the
graph, that means that the provided values are invalid (or there is a
bug in shape inference), and the result is unspecified.

Arguments:
    input (ModelProto): ModelProto

Return:
    return (ModelProto) model with inferred shape information
"""


def infer_shapes(model):  # type: (ModelProto) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('Shape inference only accepts ModelProto, '
                         'incorrect type: {}'.format(type(model)))

    model_str = model.SerializeToString()
    with open('d:/src/github/onnx.skottmckay/orig_model.txt', 'w') as f:
        f.write(str(model))
    with open('d:/src/github/onnx.skottmckay/orig_model.onnx', 'wb') as f:
        f.write(model_str)

    inferred_model_str = C.infer_shapes(model_str)

    with open('d:/src/github/onnx.skottmckay/inferred_model.onnx', 'wb') as f:
        f.write(inferred_model_str)
    inferred_model = onnx.load_from_string(inferred_model_str)
    with open('d:/src/github/onnx.skottmckay/inferred_model.txt', 'w') as f:
        f.write(str(inferred_model))

    return inferred_model
