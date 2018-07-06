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
    initial_version (String): initial opset version
    target_version (String): target opset version

Return:
    return (ModelProto) converted model

Supported adapters:
    --
"""


def convert_version(model, initial_version, target_version):  # type: (ModelProto, OpSetID, OpSetID) -> ModelProto
    if not isinstance(model, ModelProto):
        raise ValueError('VersionConverter only accepts ModelProto, incorrect type: {}'.format(type(model)))

    model_str = model.SerializeToString()
    converted_model_str = C.convert_version(model_str, initial_version, target_version)
    return onnx.load_from_string(converted_model_str)
