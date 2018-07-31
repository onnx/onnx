"""onnx version converter

This enables users to convert their models between different opsets within the
default domain ("" or "ai.onnx").
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.version_converter as C
from onnx import ModelProto
from typing import Text, Sequence

"""Apply the version conversion on the serialized ModelProto.

Arguments:
    input (ModelProto): model
    target_version (int): target opset version

Return:
    return (ModelProto) converted model

Raises Exceptions:
    RuntimeError when some necessary conversion is not supported

Supported adapters:
    --Add from Opset 7 to Opset 6
    --Add from Opset 6 to Opset 5
    --Add from Opset 6 to Opset 7
    --Add from Opset 5 to Opset 6
    --Relu from Opset 6 to Opset 5
    --Relu from Opset 5 to Opset 6
    --BatchNorm from Opset 7 to Opset 6
    --BatchNorm from Opset 6 to Opset 7
    --BatchNorm from Opset 6 to Opset 5
    --BatchNorm from Opset 5 to Opset 6
    --Concat from Opset 4 to Opset 3
    --Concat from Opset 3 to Opset 4
    --MaxPool from Opset 8 to Opset 7
    --MaxPool from Opset 7 to Opset 8
    --Reshape from Opset 5 to Opset 4
    --Reshape from Opset 4 to Opset 5
    --Sum from Opset 7 to Opset 8
    --Sum from Opset 8 to Opset 7
    --Sum from Opset 6 to Opset 5
    --Sum from Opset 5 to Opset 6
    --Gemm from Opset 7 to Opset 6
    --Gemm from Opset 6 to Opset 5
    --Gemm from Opset 6 to Opset 7
    --Gemm from Opset 5 to Opset 6
    --AveragePool from Opset 7 to Opset 6
    --AveragePool from Opset 6 to Opset 7
    --Dropout from Opset 7 to Opset 6
    --Dropout from Opset 6 to Opset 5
    --Dropout from Opset 6 to Opset 7
    --Dropout from Opset 5 to Opset 6

Unsupported adapters:
    --Any Ops not mentioned above between any opsets with breaking changes
"""


def convert_version(model, target_version):  # type: (ModelProto, int) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('VersionConverter only accepts ModelProto as model, incorrect type: {}'.format(type(model)))
    if not isinstance(target_version, int):
        raise ValueError('VersionConverter only accepts int as target_version, incorrect type: {}'.format(type(target_version)))
    model_str = model.SerializeToString()
    converted_model_str = C.convert_version(model_str, target_version)
    return onnx.load_from_string(converted_model_str)
