# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops.aionnx_preview.op_flex_attention import (
    FlexAttention,
)


def load_op(domain: str, op_type: str, version: int | None = None):  # noqa: ARG001
    """Loads the registered operator for the specified domain and type.

    Args:
        domain: operator domain
        op_type: operator type
        version: requested version

    Returns:
        class
    """
    if domain != "ai.onnx.preview":
        raise ValueError(f"Domain must be 'ai.onnx.preview' but got '{domain}'")

    ops = {
        "FlexAttention": FlexAttention,
    }

    if op_type not in ops:
        raise NotImplementedError(
            f"Op type '{op_type}' is not implemented in domain '{domain}'"
        )

    return ops[op_type]
