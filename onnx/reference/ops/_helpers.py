# SPDX-License-Identifier: Apache-2.0
from onnx.reference.op_run import (
    OpFunction,
    OpRun,
    RuntimeContextError,
    RuntimeImplementationError,
    _split_class_name,
)


def _build_registered_operators_any_domain(clo, load_op):  # type: ignore
    reg_ops = {}  # type: ignore
    for class_name, class_type in clo.items():
        if class_name[0] == "_" or class_name in {
            "Any",
            "cl",
            "clo",
            "class_name",
            "get_schema",
            "List",
            "textwrap",
            "Union",
        }:
            continue  # pragma: no cover
        if isinstance(class_type, type(load_op)):
            continue
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
    if len(reg_ops) == 0:
        raise RuntimeError("No registered operators. The installation went wrong.")
    # Set default implementation to the latest one.
    for _, impl in reg_ops.items():
        if None in impl:
            # default already exists
            continue
        max_version = max(impl)
        impl[None] = impl[max_version]
    return reg_ops
