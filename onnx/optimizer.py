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
    names (list of string): list of optimization names.

Return:
    return (ModelProto) optimized model

Supported pass names:
    -- eliminate_identity
    -- eliminate_nop_transpose
    -- eliminate_unused_initializer
    -- extract_constant_to_initializer
    -- fuse_add_bias_into_conv
    -- fuse_bn_into_conv
    -- fuse_consecutive_squeezes
    -- fuse_consecutive_transposes
    -- fuse_transpose_into_gemm
    -- lift_lexical_references
    -- split

Most passes are on by default.  See `default_enabled_passes` in
`optimizer.py` for the full list.

A few passes are off by default, because they have some unusual
behavior.  These are

    -- split. This optimization splits a single graph into two.

A note on motivations: the intent here is to behave somewhat
analogously to `gcc -O2`.  You don't have to invoke optimization, and
you are free to invoke optimization with a carefully selected set of
passes, but the default is to apply all reasonable passes, which may
grow/change over time.

"""
default_enabled_passes = [
    'eliminate_identity',
    'eliminate_nop_transpose',
    'eliminate_unused_initializer',
    'extract_constant_to_initializer',
    'fuse_add_bias_into_conv',
    'fuse_bn_into_conv',
    'fuse_consecutive_squeezes',
    'fuse_consecutive_transposes',
    'fuse_transpose_into_gemm',
    'lift_lexical_references'
]

def optimize(model, passes=[]):  # type: (ModelProto, Sequence[Text]) -> ModelProto
    passes = passes or default_enabled_passes
    if not isinstance(model, ModelProto):
        raise ValueError('Optimizer only accepts ModelProto, incorrect type: {}'.format(type(model)))

    model_str = model.SerializeToString()
    optimized_model_str = C.optimize(model_str, passes)
    return onnx.load_from_string(optimized_model_str)
