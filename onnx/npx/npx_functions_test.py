# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np

from onnx.npx.npx_core_api import (
    cst,
    make_tuple,
    npxapi_function,
    npxapi_inline,
    tuple_var,
    var,
)
from onnx.npx.npx_types import (
    ElemType,
    OptParType,
    ParType,
    SequenceType,
    TensorType,
    TupleType,
)


@npxapi_function
def _min_max(
    x: TensorType[ElemType.numerics, "T"]
) -> TupleType[TensorType[ElemType.numerics, "T"], TensorType[ElemType.numerics, "T"]]:
    "See :func:`numpy.abs`."
    return tuple_var(var(x, op="ReduceMin"), var(x, op="ReduceMax"))


@npxapi_inline
def _min_max_inline(
    x: TensorType[ElemType.numerics, "T"]
) -> TupleType[TensorType[ElemType.numerics, "T"], TensorType[ElemType.numerics, "T"]]:
    "See :func:`numpy.abs`."
    return tuple_var(var(x, op="ReduceMin"), var(x, op="ReduceMax"))


@npxapi_function
def absolute(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.abs`."
    return var(x, op="Abs")


@npxapi_function
def addition(
    x: TensorType[ElemType.numerics, "T"], y: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.addition`."
    return var(x, y, op="Add")


@npxapi_function
def argmin(
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op="ArgMin", axis=axis, keepdims=keepdims)


@npxapi_function
def concat(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]], axis: ParType[int] = 0
) -> TensorType[ElemType.numerics, "T"]:
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=axis)


@npxapi_function
def copy(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    return var(x, op="Identity")


@npxapi_function
def log1p(x: TensorType[ElemType.floats, "T"]) -> TensorType[ElemType.floats, "T"]:
    "See :func:`numpy.log1p`."
    x1 = var(x, var(cst(np.array([1], dtype=np.int64)), x, op="CastLike"), op="Add")
    return var(x1, op="Log")


@npxapi_function
def negative(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.abs`."
    return var(x, op="Neg")


@npxapi_function
def relu(
    x: TensorType[ElemType.numerics, "T"],
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.addition`."
    return var(var(absolute(x), x, op="Add"), var(cst(2), x, op="CastLike"), op="Div")


@npxapi_function
def topk(
    x: TensorType[ElemType.numerics, "T"],
    k: TensorType[ElemType.int64, "I", (1,)],
    axis: OptParType[int] = -1,
    largest: OptParType[int] = 1,
    sorted: OptParType[int] = 1,  # pylint: disable=redefined-builtin
) -> TupleType[TensorType[ElemType.numerics, "T"], TensorType[ElemType.int64, "I"]]:
    "See :func:`numpy.argsort`."
    return make_tuple(2, x, k, op="TopK", axis=axis, largest=largest, sorted=sorted)


@npxapi_function
def transpose(
    x: TensorType[ElemType.numerics, "T"], perm: ParType[Tuple[int]] = (1, 0)
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.transpose`."
    return var(x, op="Transpose", perm=list(perm))
