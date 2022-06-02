# SPDX-License-Identifier: Apache-2.0
"""onnx shape inference. Shape inference is not guaranteed to be
complete.

"""

import onnx
import onnx.onnx_cpp2py_export.shape_inference as C
from onnx import ModelProto
from typing import Union


def infer_shapes(model: Union[ModelProto, bytes], check_type: bool = False, strict_mode: bool = False, data_prop: bool = False) -> ModelProto:
    """Apply shape inference to the provided ModelProto.

    Inferred shapes are added to the value_info field of the graph.

    If the inferred values conflict with values already provided in the
    graph, that means that the provided values are invalid (or there is a
    bug in shape inference), and the result is unspecified.

    Arguments:
        model (Union[ModelProto, bytes], bool, bool, bool) -> ModelProto
        check_type (bool): Checks the type-equality for input and output
        strict_mode (bool): Stricter shape inference, it will throw errors if any;
            Otherwise, simply stop if any error
        data_prop (bool): Enables data propagation for limited operators to perform shape computation

    Returns:
        (ModelProto) model with inferred shape information
    """
    if isinstance(model, (ModelProto, bytes)):
        model_str = model if isinstance(model, bytes) else model.SerializeToString()
        inferred_model_str = C.infer_shapes(model_str, check_type, strict_mode, data_prop)
        return onnx.load_from_string(inferred_model_str)
    elif isinstance(model, str):
        raise TypeError('infer_shapes only accepts ModelProto or bytes,'
                        'you can use infer_shapes_path for the model path (String).')
    else:
        raise TypeError('infer_shapes only accepts ModelProto or bytes, '
                         'incorrect type: {}'.format(type(model)))


def infer_shapes_path(model_path: str, output_path: str = '', check_type: bool = False, strict_mode: bool = False, data_prop: bool = False) -> None:
    """
    Take model path for shape_inference same as infer_shape; it support >2GB models
    Directly output the inferred model to the output_path; Default is the original model path
    """
    if isinstance(model_path, ModelProto):
        raise TypeError('infer_shapes_path only accepts model Path (String),'
                        'you can use infer_shapes for the ModelProto.')
    # Directly output the inferred model into the specified path, return nothing
    elif isinstance(model_path, str):
        # If output_path is not defined, default output_path would be the original model path
        if output_path == '':
            output_path = model_path
        C.infer_shapes_path(model_path, output_path, check_type, strict_mode, data_prop)
    else:
        raise TypeError('infer_shapes_path only accepts model path (String), '
                         'incorrect type: {}'.format(type(model_path)))


InferenceError = C.InferenceError
