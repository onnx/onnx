# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict


from onnx.reference.op_run import OpRun, _split_class_name


def build_registered_operators_any_domain(
    module_context: Dict[str, Any]
) -> Dict[str, Dict[int, OpRun]]:

    reg_ops: Dict[str, Dict[int, OpRun]] = {}  # type: ignore
    for class_name, class_type in module_context.items():
        if class_name.startswith("_") or class_name in {
            "Any",
            "Dict",
            "List",
            "Union",
            "cl",
            "class_name",
            "get_schema",
            "module_context",
            "textwrap",
        }:
            continue
        if isinstance(class_type, type(build_registered_operators_any_domain)):
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
    if not reg_ops:
        raise RuntimeError(
            f"No registered operator. This error happens when no implementation "
            f"of type {type(impl_type)} was detected. It is suggested to reinstall the package."
        )
    # Set default implementation to the latest one.
    for impl in reg_ops.values():
        if None in impl:
            # default already exists
            continue
        max_version = max(impl)
        impl[None] = impl[max_version]
    return reg_ops
