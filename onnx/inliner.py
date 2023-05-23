# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.onnx_cpp2py_export.inliner as C  # noqa: N812


def inline_local_functions(
    model: onnx.ModelProto, convert_version: bool = False
) -> onnx.ModelProto:
    """Inline model-local functions in given model.

    Arguments:
        model: an ONNX ModelProto
        convert_version: if true, try to apply automatic version-conversion to functions requiring a
            different (ONNX) opset version from the model.
    Returns:
        ModelProto with all calls to model-local functions inlined (recursively)
    """
    result = C.inline_local_functions(model.SerializeToString(), convert_version)
    inlined_model = onnx.ModelProto()
    inlined_model.ParseFromString(result)
    return inlined_model
