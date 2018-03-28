# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx optimizer

This enables users to optimize their models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.onnx_cpp2py_export.optimizer as C

"""Apply the optimization on the serialized ModelProto.

Arguments:
    input (string): serialized ModelProto
    names (list of string): list of optimization names

Supported pass names:
    -- nop
    -- eliminate_nop_transpose
    -- fuse_consecutive_transposes
    -- fuse_transpose_into_gemm
"""
optimize = C.optimize
