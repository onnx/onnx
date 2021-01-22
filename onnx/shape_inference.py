# SPDX-License-Identifier: Apache-2.0

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
from six import string_types
from typing import Text

"""Apply shape inference to the provided ModelProto.

Inferred shapes are added to the value_info field of the graph.

If the inferred values conflict with values already provided in the
graph, that means that the provided values are invalid (or there is a
bug in shape inference), and the result is unspecified.

Arguments:
    input (Union[ModelProto, Text], Text, bool) -> ModelProto

Return:
    return (ModelProto) model with inferred shape information
"""


def infer_shapes(model, check_type=False, strict_mode=False):  # type: (ModelProto, bool, bool) -> ModelProto
    if isinstance(model, ModelProto):
        model_str = model.SerializeToString()
        inferred_model_str = C.infer_shapes(model_str, check_type, strict_mode)
        return onnx.load_from_string(inferred_model_str)
    elif isinstance(model, string_types):
        raise TypeError('infer_shapes only accepts ModelProto,'
                        'you can use infer_shapes_path for the model path (String).')
    else:
        raise TypeError('infer_shapes only accepts ModelProto, '
                         'incorrect type: {}'.format(type(model)))


def infer_shapes_path(model_path, output_path='', check_type=False, strict_mode=False):  # type: (Text, Text, bool, bool) -> None
    """
    Take model path for shape_inference same as infer_shape; it support >2GB models
    Directly output the inferred model to the output_path; Default is the original model path
    """
    if isinstance(model_path, ModelProto):
        raise TypeError('infer_shapes_path only accepts model Path (String),'
                        'you can use infer_shapes for the ModelProto.')
    # Directly output the inferred model into the specified path, return nothing
    elif isinstance(model_path, string_types):
        # If output_path is not defined, default output_path would be the original model path
        if output_path == '':
            output_path = model_path
        C.infer_shapes_path(model_path, output_path, check_type, strict_mode)
    else:
        raise TypeError('infer_shapes_path only accepts model path (String), '
                         'incorrect type: {}'.format(type(model_path)))


InferenceError = C.InferenceError
