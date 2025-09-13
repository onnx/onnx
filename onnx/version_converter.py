# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""onnx version converter

This enables users to convert their models between different opsets within the
default domain ("" or "ai.onnx").
"""

from __future__ import annotations

import onnx
import onnx.onnx_cpp2py_export.version_converter as C  # noqa: N812
from onnx import ModelProto


def convert_version(model: ModelProto, target_version: int) -> ModelProto:
    """Convert opset version of the ModelProto.

    Arguments:
        model: Model.
        target_version: Target opset version.

    Returns:
        Converted model.

    Raises:
        RuntimeError when some necessary conversion is not supported.
    """
    if not isinstance(model, ModelProto):
        raise TypeError(
            f"VersionConverter only accepts ModelProto as model, incorrect type: {type(model)}"
        )
    if not isinstance(target_version, int):
        raise TypeError(
            f"VersionConverter only accepts int as target_version, incorrect type: {type(target_version)}"
        )

    # Preserve sequence-related type information that might be lost during conversion
    preserved_sequence_types = {}

    # Preserve sequence types from graph inputs
    for input_proto in model.graph.input:
        if input_proto.type.HasField('sequence_type'):
            preserved_sequence_types[input_proto.name] = input_proto.type

    # Preserve sequence types from graph outputs
    for output_proto in model.graph.output:
        if output_proto.type.HasField('sequence_type'):
            preserved_sequence_types[output_proto.name] = output_proto.type

    # Preserve sequence types from value_info
    for value_info in model.graph.value_info:
        if value_info.type.HasField('sequence_type'):
            preserved_sequence_types[value_info.name] = value_info.type

    model_str = model.SerializeToString()
    converted_model_str = C.convert_version(model_str, target_version)
    converted_model = onnx.load_from_string(converted_model_str)

    # Restore preserved sequence type information
    for name, type_proto in preserved_sequence_types.items():
        # Check if this name exists in the converted model's inputs
        for input_proto in converted_model.graph.input:
            if input_proto.name == name:
                input_proto.type.CopyFrom(type_proto)
                break

        # Check if this name exists in the converted model's outputs
        for output_proto in converted_model.graph.output:
            if output_proto.name == name:
                output_proto.type.CopyFrom(type_proto)
                break

        # Check if this name exists in the converted model's value_info
        for value_info in converted_model.graph.value_info:
            if value_info.name == name:
                value_info.type.CopyFrom(type_proto)
                break

        # If the name doesn't exist in any of the above, add it to value_info
        exists = False
        for input_proto in converted_model.graph.input:
            if input_proto.name == name:
                exists = True
                break
        for output_proto in converted_model.graph.output:
            if output_proto.name == name:
                exists = True
                break
        for value_info in converted_model.graph.value_info:
            if value_info.name == name:
                exists = True
                break

        if not exists:
            # Add as value_info
            new_vi = converted_model.graph.value_info.add()
            new_vi.name = name
            new_vi.type.CopyFrom(type_proto)

    return converted_model


ConvertError = C.ConvertError
