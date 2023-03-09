# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import numpy as np
from onnx import FunctionProto, ModelProto, NodeProto
from onnx.numpy_helper import from_array
from onnx.npx.npx_core_api import (  # pylint: disable=W0611
    cst,
    make_tuple,
    var,
    npxapi_inline,
)
from onnx.npx.npx_types import (  # pylint: disable=W0611
    ElemType,
    OptParType,
    ParType,
    SequenceType,
    TensorType,
    TupleType,
)
from onnx.npx.npx_constants import FUNCTION_DOMAIN
from onnx.npx.npx_var import Var


def _cstv(x):
    if isinstance(x, Var):
        return x
    if isinstance(x, (int, float, np.ndarray)):
        return cst(x)
    raise TypeError(f"Unexpected constant type {type(x)}.")


@npxapi_inline
def abs(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.abs`."
    return var(x, op="Abs")


@npxapi_inline
def absolute(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.abs`."
    return var(x, op="Abs")


@npxapi_inline
def arccos(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arccos`."
    return var(x, op="Acos")


@npxapi_inline
def arccosh(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arccosh`."
    return var(x, op="Acosh")


@npxapi_inline
def amax(
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.amax`.
    """
    return var(x, op="ArgMax", axis=axis, keepdims=keepdims)


@npxapi_inline
def amin(
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.amin`.
    """
    return var(x, op="ArgMin", axis=axis, keepdims=keepdims)


@npxapi_inline
def arange(
    start_or_stop: TensorType[ElemType.int64, "I", (1,)],
    stop_or_step: Optional[TensorType[ElemType.int64, "I", (1,)]] = None,
    step: Optional[TensorType[ElemType.int64, "I", (1,)]] = None,
    dtype=None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arccos`."
    if stop_or_step is None:
        v = var(
            cst(np.array(0, dtype=np.int64)),
            start_or_stop,
            cst(np.array(1, dtype=np.int64)),
            op="Range",
        )
    elif step is None:
        v = var(
            start_or_stop, stop_or_step, cst(np.array(1, dtype=np.int64)), op="Range"
        )
    else:
        v = var(start_or_stop, stop_or_step, step, op="Range")
    if dtype is not None:
        return var(v, op="Cast", to=dtype)
    return v


@npxapi_inline
def argmax(
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.amax`.
    """
    return var(x, op="ArgMax", axis=axis, keepdims=keepdims)


@npxapi_inline
def argmin(
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = 0,
    keepdims: OptParType[int] = 0,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.argmin`.
    """
    return var(x, op="ArgMin", axis=axis, keepdims=keepdims)


@npxapi_inline
def arcsin(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arcsin`."
    return var(x, op="Asin")


@npxapi_inline
def arcsinh(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arcsinh`."
    return var(x, op="Asinh")


@npxapi_inline
def arctan(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arctan`."
    return var(x, op="Atan")


@npxapi_inline
def arctanh(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.arctanh`."
    return var(x, op="Atanh")


@npxapi_inline
def cdist(
    xa: TensorType[ElemType.numerics, "T"],
    xb: TensorType[ElemType.numerics, "T"],
    metric: OptParType[str] = "euclidean",
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`scipy.special.distance.cdist`.
    """
    return var(xa, xb, op=(FUNCTION_DOMAIN, "CDist"), metric=metric)


@npxapi_inline
def ceil(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.ceil`."
    return var(x, op="Ceil")


@npxapi_inline
def clip(
    x: TensorType[ElemType.numerics, "T"],
    a_min: TensorType[ElemType.numerics, "T"] = None,
    a_max: TensorType[ElemType.numerics, "T"] = None,
):
    "See :func:`np.clip`."
    args = [x]
    if a_min is not None:
        args.append(_cstv(a_min))
    else:
        args.append(None)
    if a_max is not None:
        args.append(_cstv(a_max))
    return var(*args, op="Clip")


@npxapi_inline
def compress(
    condition: TensorType[ElemType.bool_, "B"],
    x: TensorType[ElemType.numerics, "T"],
    axis: OptParType[int] = None,
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.compress`.
    `np.compress(condition, x)` or `npnx.compress(x, condition)`.
    """
    if axis is None:
        return var(x, condition, op="Compress")
    return var(x, condition, op="Compress", axis=axis)


@npxapi_inline
def compute(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]],
    proto: ParType[Union[FunctionProto, ModelProto, NodeProto]] = None,
    name: ParType[str] = None,
) -> TupleType[TensorType[ElemType.numerics, "T"]]:
    """
    Operator concat, handle :func:`np.vstack` and
    :func:`np.hstack`.
    """
    return var(*x, op=proto, name=name)


@npxapi_inline
def concat(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]], axis: ParType[int] = 0
) -> TensorType[ElemType.numerics, "T"]:
    """
    Operator concat, handle :func:`np.vstack` and
    :func:`np.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate."
        )
    return var(*x, op="Concat", axis=axis)


@npxapi_inline
def cos(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.cos`."
    return var(x, op="Cos")


@npxapi_inline
def cosh(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.cosh`."
    return var(x, op="Cosh")


@npxapi_inline
def cumsum(
    x: TensorType[ElemType.numerics, "T"],
    axis: Optional[TensorType[ElemType.int64, "I"]] = None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.cumsum`."
    if axis is None:
        m1 = cst(np.array([-1], dtype=np.int64))
        flat = var(x, m1, op="Reshape")
        axis = cst(np.array([0], dtype=np.int64))
        return var(flat, axis, op="CumSum")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="CumSum")


@npxapi_inline
def det(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.linalg:det`."
    return var(x, op="Det")


@npxapi_inline
def dot(
    a: TensorType[ElemType.numerics, "T"], b: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`np.dot`
    dot is equivalent to `npx.matmul == np.matmul != np.dot`
    with arrays with more than 3D dimensions.
    """
    return var(a, b, op="MatMul")


@npxapi_inline
def einsum(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]], equation: ParType[str]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.einsum`."
    return var(*x, op="Einsum", equation=equation)


@npxapi_inline
def erf(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:erf`."
    return var(x, op="Erf")


@npxapi_inline
def exp(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.exp`."
    return var(x, op="Exp")


@npxapi_inline
def expand_dims(
    x: TensorType[ElemType.numerics, "T"], axis: TensorType[ElemType.int64, "I"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.expand_dims`."
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Unsqueeze")


@npxapi_inline
def expit(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:expit`."
    return var(x, op="Sigmoid")


@npxapi_inline
def floor(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.floor`."
    return var(x, op="Floor")


@npxapi_inline
def hstack(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.hstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate."
        )
    return var(*x, op="Concat", axis=-1)


@npxapi_inline
def copy(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    return var(x, op="Identity")


@npxapi_inline
def identity(n: ParType[int], dtype=None) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    val = np.array([n, n], dtype=np.int64)
    shape = cst(val)
    model = var(
        shape, op="ConstantOfShape", value=from_array(np.array([0], dtype=np.int64))
    )
    v = var(model, dtype=dtype, op="EyeLike")
    return v


@npxapi_inline
def isnan(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.bool_, "T"]:
    "See :func:`np.isnan`."
    return var(x, op="IsNaN")


@npxapi_inline
def log(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.log`."
    return var(x, op="Log")


@npxapi_inline
def log1p(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.log1p`."
    x1 = var(x, var(cst(np.array([1])), x, op="CastLike"), op="Add")
    return var(x1, op="Log")


@npxapi_inline
def matmul(
    a: TensorType[ElemType.numerics, "T"], b: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.matmul`."
    return var(a, b, op="MatMul")


@npxapi_inline
def pad(
    x: TensorType[ElemType.numerics, "T"],
    pads: TensorType[ElemType.int64, "I"],
    constant_value: Optional[TensorType[ElemType.numerics, "T"]] = None,
    axes: Optional[TensorType[ElemType.int64, "I"]] = None,
    mode: ParType[str] = "constant",
):
    """
    It does not implement :func:`np.pad` but the ONNX version.
    """
    if constant_value is None:
        if axes is None:
            return var(x, pads, op="Pad", mode=mode)
        return var(x, pads, None, axes, op="Pad", mode=mode)
    if axes is None:
        return var(x, pads, constant_value, op="Pad", mode=mode)
    return var(x, pads, constant_value, axes, op="Pad", mode=mode)


@npxapi_inline
def reciprocal(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.reciprocal`."
    return var(x, op="Reciprocal")


@npxapi_inline
def relu(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "relu"
    return var(x, op="Relu")


@npxapi_inline
def round(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.round`."
    return var(x, op="Round")


@npxapi_inline
def sigmoid(
    x: TensorType[ElemType.numerics, "T"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:expit`."
    return var(x, op="Sigmoid")


@npxapi_inline
def sign(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.sign`."
    return var(x, op="Sign")


@npxapi_inline
def sin(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.sin`."
    return var(x, op="Sin")


@npxapi_inline
def sinh(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.sinh`."
    return var(x, op="Sinh")


@npxapi_inline
def sqrt(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.sqrt`."
    return var(x, op="Sqrt")


@npxapi_inline
def squeeze(
    x: TensorType[ElemType.numerics, "T"],
    axis: Optional[TensorType[ElemType.int64, "I"]] = None,
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.squeeze`."
    if axis is None:
        shape = x.shape
        zero = cst(np.array([0], dtype=np.int64))
        one = cst(np.array([1], dtype=np.int64))
        ind = var(zero, shape.shape, one, op="Range")
        axis = var(ind, shape == one, op="Compress")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Squeeze")


@npxapi_inline
def tan(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.tan`."
    return var(x, op="Tan")


@npxapi_inline
def tanh(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.tanh`."
    return var(x, op="Tanh")


@npxapi_inline
def topk(
    x: TensorType[ElemType.numerics, "T"],
    k: TensorType[ElemType.int64, "I", (1,)],
    axis: OptParType[int] = -1,
    largest: OptParType[int] = 1,
    sorted: OptParType[int] = 1,
) -> TupleType[TensorType[ElemType.numerics, "T"], TensorType[ElemType.int64, "I"]]:
    "See :func:`np.argsort`."
    return make_tuple(2, x, k, op="TopK", axis=axis, largest=largest, sorted=sorted)


@npxapi_inline
def transpose(
    x: TensorType[ElemType.numerics, "T"], perm: ParType[Tuple[int, ...]] = (1, 0)
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.transpose`."
    return var(x, op="Transpose", perm=list(perm))


@npxapi_inline
def unsqueeze(
    x: TensorType[ElemType.numerics, "T"], axis: TensorType[ElemType.int64, "I"]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.expand_dims`."
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(np.array(axis, dtype=np.int64))
    return var(x, axis, op="Unsqueeze")


@npxapi_inline
def vstack(
    *x: SequenceType[TensorType[ElemType.numerics, "T"]]
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.vstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate."
        )
    return var(*x, op="Concat", axis=0)


@npxapi_inline
def where(
    cond: TensorType[ElemType.bool_, "B"],
    x: TensorType[ElemType.numerics, "T"],
    y: TensorType[ElemType.numerics, "T"],
) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`np.where`."
    return var(cond, x, y, op="Where")
