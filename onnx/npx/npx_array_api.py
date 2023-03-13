# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np

from onnx.npx.npx_types import OptParType, ParType, TupleType


class ArrayApi:
    """
    List of supported method by a tensor.
    """

    def generic_method(self, method_name, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Method {method_name!r} must be overwritten for class {self.__class__.__name__!r}. "
            f"Method 'generic_method' can be overwritten as well to change the behaviour "
            f"for all methods supported by class ArrayApi."
        )

    def numpy(self) -> np.ndarray:
        return self.generic_method("numpy")

    def __neg__(self) -> "ArrayApi":
        return self.generic_method("__neg__")

    def __invert__(self) -> "ArrayApi":
        return self.generic_method("__invert__")

    def __add__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__add__", ov)

    def __radd__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__radd__", ov)

    def __sub__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__sub__", ov)

    def __rsub__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rsub__", ov)

    def __mul__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__mul__", ov)

    def __rmul__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rmul__", ov)

    def __matmul__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__matmul__", ov)

    def __truediv__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__truediv__", ov)

    def __rtruediv__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rtruediv__", ov)

    def __mod__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__mod__", ov)

    def __rmod__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rmod__", ov)

    def __pow__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__pow__", ov)

    def __rpow__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rpow__", ov)

    def __lt__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__lt__", ov)

    def __le__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__le__", ov)

    def __gt__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__gt__", ov)

    def __ge__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__ge__", ov)

    def __eq__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__eq__", ov)

    def __ne__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__ne__", ov)

    def __lshift__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__lshift__", ov)

    def __rshift__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rshift__", ov)

    def __and__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__and__", ov)

    def __rand__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rand__", ov)

    def __or__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__or__", ov)

    def __ror__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__ror__", ov)

    def __xor__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__xor__", ov)

    def __rxor__(self, ov: "ArrayApi") -> "ArrayApi":
        return self.generic_method("__rxor__", ov)

    @property
    def T(self) -> "ArrayApi":
        return self.generic_method("T")

    def astype(self, dtype: Any) -> "ArrayApi":
        return self.generic_method("astype", dtype)

    @property
    def shape(self) -> "ArrayApi":
        return self.generic_method("shape")

    def reshape(self, shape: "ArrayApi") -> "ArrayApi":
        return self.generic_method("reshape", shape)

    def sum(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "ArrayApi":
        return self.generic_method("sum", axis=axis, keepdims=keepdims)

    def mean(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "ArrayApi":
        return self.generic_method("mean", axis=axis, keepdims=keepdims)

    def min(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "ArrayApi":
        return self.generic_method("min", axis=axis, keepdims=keepdims)

    def max(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "ArrayApi":
        return self.generic_method("max", axis=axis, keepdims=keepdims)

    def prod(
        self, axis: OptParType[TupleType[int]] = None, keepdims: ParType[int] = 0
    ) -> "ArrayApi":
        return self.generic_method("prod", axis=axis, keepdims=keepdims)

    def copy(self) -> "ArrayApi":
        return self.generic_method("copy")

    def flatten(self) -> "ArrayApi":
        return self.generic_method("flatten")

    def __getitem__(self, index: Any) -> "ArrayApi":
        return self.generic_method("__getitem__", index)

    def __setitem__(self, index: Any, values: Any):
        return self.generic_method("__setitem__", index, values)
