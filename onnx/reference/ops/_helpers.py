# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

from onnx.reference.op_run import OpRun


def _get_axis(axis: int | np.ndarray) -> int:
    """Returns ``axis`` as an integer for operators expecting a scalar axis input.

    The specification asks for a 0-D tensor but ONNX Runtime, TVM and the ONNX
    shape inference also accept a 1-element 1-D tensor, so both are allowed here.
    """
    arr = np.asarray(axis)
    if arr.ndim > 1 or arr.size != 1:
        raise ValueError(f"Axis must be a scalar tensor, got shape {arr.shape}.")
    return int(arr.item())


def _split_class_name(name):
    if "_" in name:
        prefix, vers = name.rsplit("_", maxsplit=1)
        try:
            v = int(vers)
        except ValueError:
            return name, None
        return prefix, v
    return name, None


def build_registered_operators_any_domain(
    module_context: dict[str, Any],
) -> dict[str, dict[int | None, type[OpRun]]]:
    reg_ops: dict[str, dict[int | None, type[OpRun]]] = {}
    for class_name, class_type in module_context.items():
        if class_name.startswith("_") or class_name in {
            "Any",
            "Dict",
            "List",
            "TOptional",
            "Union",
            "annotations",
            "cl",
            "class_name",
            "get_schema",
            "module_context",
            "textwrap",
        }:
            continue
        if isinstance(class_type, type(build_registered_operators_any_domain)):
            continue
        issub: bool = False
        try:
            issub = issubclass(class_type, OpRun)
        except TypeError as e:
            raise TypeError(
                f"Unexpected variable type {class_type!r} and class_name={class_name!r}."
            ) from e
        if issub:
            op_type, op_version = _split_class_name(class_name)
            if op_type not in reg_ops:
                reg_ops[op_type] = {}
            reg_ops[op_type][op_version] = class_type
    if not reg_ops:
        raise RuntimeError(
            "No registered operator. This error happens when no implementation "
            "of type 'OpRun' was detected. It may be due to an error during installation. Please try reinstalling onnx."
        )
    # Set default implementation to the latest one.
    for impl in reg_ops.values():
        if None in impl:
            # default already exists
            continue
        max_version = max(impl)  # type: ignore[type-var]
        impl[None] = impl[max_version]
    return reg_ops
