# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

__all__ = ["ReferenceEvaluator", "astype"]

from onnx.reference.reference_evaluator import ReferenceEvaluator


def astype(tensor, dtype):
    """Cast a tensor whether it is a numpy array or a pytorch tensor."""
    if hasattr(tensor, "astype"):
        return tensor.astype(dtype)
    return tensor.to(dtype)


def apimod(tensor):
    """Return the module responsible for that tensor."""
    if hasattr(tensor, "astype"):
        import numpy

        return numpy
    import torch

    return torch
