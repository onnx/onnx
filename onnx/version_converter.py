"""onnx version converter

This enables users to convert their models between different opsets within
the default domain ("" or "ai.onnx").
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.version_converter as C
from onnx import ModelProto, OperatorSetIdProto
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
    --
"""


def convert_version(model, target_version):  # type: (ModelProto, int) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('VersionConverter only accepts ModelProto as model, incorrect type: {}'.format(type(model)))
    if not isinstance(target_version, int):
        raise ValueError('VersionConverter only accepts int as target_version, incorrect type: {}'.format(type(target_version)))
    model_str = model.SerializeToString()
    converted_model_str = C.convert_version(model_str, target_version)
    return onnx.load_from_string(converted_model_str)
