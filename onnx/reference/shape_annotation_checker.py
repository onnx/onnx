# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Optional runtime validation of static shape annotations.

This module implements the runtime semantics described in
``docs/ShapeAnnotationSemantics.md``: each symbolic dimension name
(``dim_param``) occurring in a model is treated as existentially quantified
once per inference run. A :class:`SymbolBindings` instance is a partial
map from such names to the (non-negative integer) value they were first
observed to take during a run; every later occurrence of the same name is
checked against that first binding.

As a convenience, alongside shape checking this module also validates a
tensor-typed value's element type (``elem_type``) against its declared
``TensorProto.DataType``, since this is a simple, purely local check (it
does not involve symbolic dimensions or bindings) that ONNX's static
checker already performs, but which is otherwise easy to violate
undetected at runtime (e.g. a kernel producing the wrong dtype).

This module only supports today's single, global namespace of symbolic
dimension names (a name means the same thing everywhere it occurs in a
model, including inside nested subgraphs) and only validates tensor-typed
values. Locally scoped names (for example, a name local to one iteration of
a ``Loop`` body, or to one element of a ``Sequence``) are not addressed; see
``docs/ShapeAnnotationSemantics.md`` for context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from onnx.helper import tensor_dtype_to_np_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from onnx.onnx_pb import TensorShapeProto, TypeProto


class ShapeAnnotationError(RuntimeError):
    """Raised when a value's actual runtime shape violates its declared static shape annotation."""


class SymbolBindings:
    """Tracks the runtime value bound to each symbolic dimension name (``dim_param``) during one inference run.

    A single instance of this class should be used for exactly one call to
    :meth:`onnx.reference.ReferenceEvaluator.run`, since a symbolic
    dimension name is existentially quantified once per run: different runs
    of the same model may bind the same name to different values.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, int] = {}

    def check_and_bind(
        self, dim: TensorShapeProto.Dimension, actual: int, *, context: str
    ) -> None:
        """Checks a single dimension against an actual (runtime) axis size, binding it if unbound.

        Args:
            dim: the declared ``Dimension`` (``dim_value``, ``dim_param``, or neither).
            actual: the actual runtime size of the corresponding axis.
            context: a human-readable description of the value and axis being checked, used
                only for error messages.

        Raises:
            ShapeAnnotationError: if ``dim`` has a ``dim_value`` that disagrees with ``actual``,
                or a ``dim_param`` already bound to a different value.
        """
        which = dim.WhichOneof("value")
        if which == "dim_value":
            if dim.dim_value != actual:
                raise ShapeAnnotationError(
                    f"{context}: declared dimension value {dim.dim_value} does not match "
                    f"actual dimension value {actual}."
                )
        elif which == "dim_param":
            name = dim.dim_param
            bound = self._bindings.get(name)
            if bound is None:
                self._bindings[name] = actual
            elif bound != actual:
                raise ShapeAnnotationError(
                    f"{context}: symbolic dimension {name!r} was previously bound to {bound}, "
                    f"but this value has actual dimension value {actual}."
                )
        # An unset oneof (neither dim_value nor dim_param) is an anonymous unknown
        # dimension: it asserts nothing and is always satisfied.


def check_shape(
    shape_proto: TensorShapeProto | None,
    actual_shape: Sequence[int],
    bindings: SymbolBindings,
    name: str,
) -> None:
    """Checks an actual runtime tensor shape against a declared static shape annotation.

    This implements the per-value check described in
    ``docs/ShapeAnnotationSemantics.md``: a missing ``shape_proto`` (unknown rank)
    is always satisfied; a rank mismatch always fails; otherwise each axis is
    checked (and, for symbolic dimensions, bound) via ``bindings``.

    Args:
        shape_proto: the declared ``TensorShapeProto``, or ``None`` if the value's
            type declares no shape (unknown rank).
        actual_shape: the actual runtime shape of the value.
        bindings: the :class:`SymbolBindings` for the current inference run.
        name: the name of the value being checked, used only for error messages.

    Raises:
        ShapeAnnotationError: if the actual shape does not satisfy the annotation.
    """
    if shape_proto is None:
        return
    declared_rank = len(shape_proto.dim)
    actual_rank = len(actual_shape)
    if declared_rank != actual_rank:
        raise ShapeAnnotationError(
            f"Value {name!r}: declared shape annotation has rank {declared_rank}, "
            f"but the actual value has rank {actual_rank} (shape {tuple(actual_shape)})."
        )
    for axis, (dim, actual) in enumerate(
        zip(shape_proto.dim, actual_shape, strict=False)
    ):
        bindings.check_and_bind(
            dim, int(actual), context=f"Value {name!r}, axis {axis}"
        )


def check_value_against_type(
    type_proto: TypeProto | None,
    value: Any,
    bindings: SymbolBindings,
    name: str,
) -> None:
    """Checks a runtime value against its declared ``TypeProto``, if it declares a tensor shape.

    Only tensor-typed values with a ``.shape`` attribute (e.g. a NumPy array) are
    checked in this initial implementation; other value kinds (sequences, maps,
    optionals, or a missing/absent type) are silently skipped. See the module
    docstring for the scope of what this checks.

    Args:
        type_proto: the value's declared ``TypeProto``, or ``None`` if not known.
        value: the actual runtime value produced during execution.
        bindings: the :class:`SymbolBindings` for the current inference run.
        name: the name of the value being checked, used only for error messages.

    Raises:
        ShapeAnnotationError: if the value's shape or element type does not
            match its declared type.
    """
    if type_proto is None or not type_proto.HasField("tensor_type"):
        return
    shape = getattr(value, "shape", None)
    if shape is None:
        # Not an array-like value (e.g. produced by an op whose output does not
        # match its declared type); nothing to check here.
        return
    tensor_type = type_proto.tensor_type
    if tensor_type.elem_type:  # 0 means UNDEFINED, i.e. no declared element type.
        actual_dtype = getattr(value, "dtype", None)
        if actual_dtype is not None:
            try:
                declared_dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type)
            except KeyError:
                # No corresponding numpy dtype (e.g. a storage-only element
                # type); nothing to compare against.
                declared_dtype = None
            # The reference evaluator represents ONNX STRING tensors as
            # NumPy Unicode arrays, while the dtype mapping uses object.
            string_unicode = (
                declared_dtype == np.dtype(object)
                and getattr(actual_dtype, "kind", None) == "U"
            )
            if (
                declared_dtype is not None
                and actual_dtype != declared_dtype
                and not string_unicode
            ):
                raise ShapeAnnotationError(
                    f"Value {name!r}: declared element type {declared_dtype} does not "
                    f"match actual element type {actual_dtype}."
                )
    shape_proto = tensor_type.shape if tensor_type.HasField("shape") else None
    check_shape(shape_proto, shape, bindings, name)
