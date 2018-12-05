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
from typing import Text, Sequence, Optional

"""Apply the optimization on the serialized ModelProto.

Arguments:
    input (ModelProto): model
    names (list of string): list of optimization names

Return:
    return (ModelProto) optimized model

Supported pass names:
    -- nop
    -- eliminate_deadend
    -- eliminate_identity
    -- eliminate_nop_dropout
    -- eliminate_nop_monotone_argmax
    -- eliminate_nop_pad
    -- eliminate_nop_transpose
    -- eliminate_unused_initializer
    -- extract_constant_to_initializer
    -- fuse_add_bias_into_conv
    -- fuse_bn_into_conv
    -- fuse_consecutive_concats
    -- fuse_consecutive_log_softmax
    -- fuse_consecutive_reduce_unsqueeze
    -- fuse_consecutive_squeezes
    -- fuse_consecutive_transposes
    -- fuse_matmul_add_bias_into_gemm
    -- fuse_pad_into_conv
    -- fuse_transpose_into_gemm
"""

get_available_passes = C.get_available_passes


def optimize(model, passes=None, fixed_point=False):  # type: (ModelProto, Optional[Sequence[Text]], bool) -> ModelProto
    if passes is None:
        passes = ['eliminate_nop_transpose',
                  'eliminate_nop_pad',
                  'fuse_consecutive_transposes',
                  'fuse_transpose_into_gemm']

    if not isinstance(model, ModelProto):
        raise ValueError(
            'Optimizer only accepts ModelProto, incorrect type: {}'.format(type(model)))

    model_str = model.SerializeToString()
    if fixed_point:
        optimized_model_str = C.optimize_fixedpoint(model_str, passes)
    else:
        optimized_model_str = C.optimize(model_str, passes)

    return onnx.load_from_string(optimized_model_str)
