# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops.aionnx_preview._op_list import (
    load_op as _load_op,
)

def load_op(domain: str, op_type: str, version: int | None = None):
    """
    Loads the registered operator for the specified domain and type.
    
    Args:
        domain: operator domain
        op_type: operator type
        version: requested version
        
    Returns:
        class
    """
    return _load_op(domain, op_type, version=version)
