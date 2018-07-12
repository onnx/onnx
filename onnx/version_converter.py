"""onnx version converter

This enables users to convert their models between different opsets.
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
    initial_version (OperatorSetIdProto): initial opset version
    target_version (OperatorSetIdProto): target opset version

Return:
    return (ModelProto) converted model

Raises Exceptions:
    RuntimeError when some necessary conversion is not supported

Supported adapters:
    --
"""


def convert_version(model, initial_version, target_version):  # type: (ModelProto, OperatorSetIdProto, OperatorSetIdProto) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('VersionConverter only accepts ModelProto as model, incorrect type: {}'.format(type(model)))
    if not isinstance(initial_version, OperatorSetIdProto):
        raise ValueError('VersionConverter only accepts OperatorSetIdProto as initial_version, incorrect type: {}'.format(type(initial_version)))
    if not isinstance(target_version, OperatorSetIdProto):
        raise ValueError('VersionConverter only accepts OperatorSetIdProto as target_version, incorrect type: {}'.format(type(target_version)))
    model_str = model.SerializeToString()
    initial_str = initial_version.SerializeToString()
    target_str = target_version.SerializeToString()
    converted_model_str = C.convert_version(model_str, initial_str, target_str)
    return onnx.load_from_string(converted_model_str)
