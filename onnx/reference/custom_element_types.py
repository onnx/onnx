# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings

from onnx.custom_element_types import bfloat16
from onnx.custom_element_types import float8e4m3fn
from onnx.custom_element_types import float8e4m3fnuz
from onnx.custom_element_types import float8e5m2
from onnx.custom_element_types import float8e5m2fnuz
from onnx.custom_element_types import int4
from onnx.custom_element_types import uint4

warnings.warn(
    "This file was moved to onnx.custom_element_types. Importing this module will fail in onnx>=1.19.0.",
    category=PendingDeprecationWarning,
    stacklevel=1,
)


__all__ = [
    "bfloat16",
    "float8e4m3fn",
    "float8e4m3fnuz",
    "float8e5m2",
    "float8e5m2fnuz",
    "int4",
    "uint4",
]
