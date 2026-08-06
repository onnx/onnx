# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def _check_gated_inputs(op_type: str, a: np.ndarray, b: np.ndarray) -> None:
    # The two-input gated activations (SwiGLU, GeGLU) require identical shapes and
    # dtypes for A and B: broadcasting is not applied, matching the
    # equal-shape/no-broadcast contract enforced by GatedActivationShapeInference
    # at graph-build time.
    if a.shape != b.shape:
        raise ValueError(
            f"{op_type} requires inputs A and B to have identical shapes "
            f"(broadcasting is not applied), but got A.shape={a.shape} and "
            f"B.shape={b.shape}."
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"{op_type} requires inputs A and B to have identical dtypes, but "
            f"got A.dtype={a.dtype} and B.dtype={b.dtype}."
        )
