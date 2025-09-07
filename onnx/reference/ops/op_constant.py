# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun, RefAttrName


def _check_dtype(val):
    dtype = val.dtype
    if not isinstance(dtype, np.dtype):
        raise TypeError(
            f"Type ({dtype}, {type(dtype)}) is not a numpy type (operator 'Constant')"
        )


class ConstantCommon(OpRun):
    def _check(self, cst):
        if isinstance(cst, tuple):
            raise TypeError(f"Unexpected type {type(cst)} for a constant.")
        return cst


class Constant_1(ConstantCommon):
    def __init__(self, onnx_node, run_params):
        ConstantCommon.__init__(self, onnx_node, run_params)
        self.cst = self.value
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):
        if overridden_attributes and (
            len(overridden_attributes) > 1
            or "value" not in overridden_attributes
            or id(overridden_attributes["value"]) != id(self.value)
        ):
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_9(Constant_1):
    def __init__(self, onnx_node, run_params):
        Constant_1.__init__(self, onnx_node, run_params)


class Constant_11(ConstantCommon):
    def __init__(self, onnx_node, run_params):
        ConstantCommon.__init__(self, onnx_node, run_params)
        if getattr(self, "sparse_value", None) is None:
            self.cst = self.value
        else:
            self.cst = self.sparse_value
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):
        if overridden_attributes and (
            len(overridden_attributes) > 1
            or "value" not in overridden_attributes
            or id(overridden_attributes["value"]) != id(self.value)
        ):
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_12(ConstantCommon):
    def __init__(self, onnx_node, run_params):
        ConstantCommon.__init__(self, onnx_node, run_params)
        if hasattr(self, "sparse_value") and self.sparse_value is not None:
            self.cst_name = "sparse_value"
            self.cst = self.sparse_value
            self.cst_convert = lambda v: v
        elif hasattr(self, "value") and self.value is not None:
            self.cst_name = "value"
            self.cst = self.value
            self.cst_convert = lambda v: v
        else:
            for attr, np_dtype in {
                "value_float": np.float32,
                "value_floats": np.float32,
                "value_int": np.int64,
                "value_ints": np.int64,
                "value_string": np.str_,
                "value_strings": np.str_,
            }.items():
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    self.cst_name = attr
                    v = getattr(self, attr)
                    self.cst = (
                        v if isinstance(v, RefAttrName) else np.array(v, dtype=np_dtype)
                    )
                    self.cst_convert = lambda v, np_dtype=np_dtype: np.array(
                        v, dtype=np_dtype
                    )
                    break
        if not hasattr(self, "cst_name"):
            raise AttributeError(
                f"No constant is defined for operator 'Constant', outputs are {onnx_node.output}."
            )

    def _run(self, **overridden_attributes):
        if self.has_linked_attribute:
            if overridden_attributes is None:
                raise RuntimeError(
                    f"Attributes are empty, cannot retrieve value for {self.cst!r}."
                )
            if self.cst_name not in overridden_attributes:
                raise RuntimeError(
                    f"Cannot find attribute {self.cst_name!r} in {list(overridden_attributes)!r}."
                )
            value = overridden_attributes[self.cst_name]
            if isinstance(value, np.ndarray):
                return (value,)
            return (self.cst_convert(value),)
        return (self._check(self.cst),)
