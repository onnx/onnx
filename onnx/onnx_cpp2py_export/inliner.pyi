# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

def inline_local_functions(model: bytes, convert_version: bool) -> bytes:
    """
    Inlines calls to model-local function in input model and returns it.
    Both input and output are serialized ModelProtos.
    """
    ...
