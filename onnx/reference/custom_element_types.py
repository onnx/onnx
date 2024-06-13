# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings

from onnx.custom_element_types import bfloat16  # noqa: F401
from onnx.custom_element_types import float8e4m3fn  # noqa: F401
from onnx.custom_element_types import float8e4m3fnuz  # noqa: F401
from onnx.custom_element_types import float8e5m2  # noqa: F401
from onnx.custom_element_types import float8e5m2fnuz  # noqa: F401
from onnx.custom_element_types import int4  # noqa: F401
from onnx.custom_element_types import uint4  # noqa: F401

warnings.warn(
    "This file was moved to onnx.custom_element_types. Importing this module will fail in onnx>=1.19.0.",
    category=DeprecationWarning,
    stacklevel=1,
)
