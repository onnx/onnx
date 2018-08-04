# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx optimizer

This enables users to optimize their models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.optimizer as C
from onnx import ModelProto
from typing import Text, Sequence

"""Apply the optimization on the serialized ModelProto.

Arguments:
    input (ModelProto): model
    names (list of string): list of optimization names

Return:
    return (ModelProto) optimized model

Supported pass names:
    -- nop
    -- eliminate_identity
    -- eliminate_nop_transpose
    -- eliminate_unused_initializer
    -- fuse_consecutive_squeezes
    -- fuse_consecutive_transposes
    -- fuse_add_bias_into_conv
    -- fuse_transpose_into_gemm
"""

get_available_passes = C.get_available_passes


def optimize(model, passes=[]):  # type: (ModelProto, Sequence[Text]) -> ModelProto
    if len(passes) == 0:
        passes = ['eliminate_nop_transpose',
                  'fuse_consecutive_transposes',
                  'fuse_transpose_into_gemm']
    if not isinstance(model, ModelProto):
        raise ValueError('Optimizer only accepts ModelProto, incorrect type: {}'.format(type(model)))

    model_str = model.SerializeToString()
    optimized_model_str = C.optimize(model_str, passes)
    return onnx.load_from_string(optimized_model_str)
