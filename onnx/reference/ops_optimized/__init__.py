# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx.reference.op_run import OpRun

optimized_operators: list[type[OpRun]] = []

__all__ = ["optimized_operators"]
