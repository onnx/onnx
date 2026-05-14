# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility shim for onnx.version.

This module is deprecated. Use ``onnx.__version__`` instead.
"""

from __future__ import annotations

import warnings

from onnx import __version__ as version

warnings.warn(
    "onnx.version is deprecated. Use onnx.__version__ instead.",
    DeprecationWarning,
    stacklevel=2,
)

git_version = ""

__all__ = ["version", "git_version"]
