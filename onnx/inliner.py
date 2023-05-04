# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.onnx_cpp2py_export.inliner as C  # noqa: N812


def inline_local_functions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Inline model-local functions in given model.

    Arguments:
        model: an ONNX ModelProto
    Returns:
        ModelProto with all calls to model-local functions inlined (recursively)
    """
    result = C.inline_local_functions(model.SerializeToString())
    inlined_model = onnx.ModelProto()
    inlined_model.ParseFromString(result)
    return inlined_model
