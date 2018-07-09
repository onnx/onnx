# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx version converter

This enables users to convert their models between different opsets.
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
    initial_version (OperatorSetIdProto): initial opset version
    target_version (OperatorSetIdProto): target opset version

Return:
    return (ModelProto) converted model

Supported adapters:
    --
"""


def convert_version(model, initial_version, target_version):  # type: (ModelProto, OperatorSetIdProto, OperatorSetIdProto) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('VersionConverter only accepts ModelProto, incorrect type: {}'.format(type(model)))
    if not isinstance(initial_version, OperatorSetIdProto) or not isinstance(
            target_version, OperatorSetIdProto):
        raise ValueError('VersionConverter only accepts OperatorSetIdProto, incorrect type: {}'.format(type(model)))
    print("Converting model")
    model_str = model.SerializeToString()
    initial_str = initial_version.SerializeToString()
    target_str = target_version.SerializeToString()
    converted_model_str = C.convert_version(model_str, initial_str, target_str)
    return onnx.load_from_string(converted_model_str)
