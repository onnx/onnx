# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun, RefAttrName


def _check_dtype(val):  # type: ignore
    a = val.dtype
    if not isinstance(a, numpy.dtype) and a not in {
        numpy.int8,
        numpy.uint8,
        numpy.float16,
        numpy.float32,
        numpy.float64,
        numpy.int32,
        numpy.int64,
        numpy.int16,
        numpy.uint16,
        numpy.uint32,
        numpy.bool_,
        numpy.str_,
        numpy.uint64,
        bool,
        str,
    }:
        raise TypeError(
            f"Type ({a}, {type(a)}) is not a numpy type (operator 'Constant')"
        )


class ConstantCommon(OpRun):
    def _check(self, cst):  # type: ignore
        if isinstance(cst, tuple):
            raise TypeError(f"Unexpected type {type(cst)} for a constant.")
        return cst


class Constant_1(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        self.cst = self.value  # type: ignore
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):  # type: ignore
        if overridden_attributes:
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_9(Constant_1):
    def __init__(self, onnx_node, run_params):  # type: ignore
        Constant_1.__init__(self, onnx_node, run_params)


class Constant_11(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        if getattr(self, "sparse_value", None) is None:
            self.cst = self.value  # type: ignore
        else:
            self.cst = self.sparse_value  # type: ignore
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):  # type: ignore
        if overridden_attributes:
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_12(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        if hasattr(self, "sparse_value") and self.sparse_value is not None:  # type: ignore
            self.cst_name = "sparse_value"
            self.cst = self.sparse_value  # type: ignore
        elif hasattr(self, "value_float") and self.value_float is not None:  # type: ignore
            self.cst_name = "value_float"
            self.cst = numpy.array(self.value_float, dtype=numpy.float32)  # type: ignore
        elif hasattr(self, "value_floats") and self.value_floats is not None:  # type: ignore
            self.cst_name = "value_floats"
            self.cst = numpy.array(self.value_floats, dtype=numpy.float32)  # type: ignore
        elif hasattr(self, "value_int") and self.value_int is not None:  # type: ignore
            self.cst_name = "value_int"
            self.cst = numpy.array(self.value_int, dtype=numpy.int64)  # type: ignore
        elif hasattr(self, "value_ints") and self.value_ints is not None:  # type: ignore
            self.cst_name = "value_ints"
            self.cst = numpy.array(self.value_ints, dtype=numpy.int64)  # type: ignore
        elif hasattr(self, "value_string") and self.value_string is not None:  # type: ignore
            self.cst_name = "value_string"
            self.cst = numpy.array(self.value_string)  # type: ignore
        elif hasattr(self, "value_strings") and self.value_strings is not None:  # type: ignore
            self.cst_name = "value_strings"
            self.cst = numpy.array(self.value_strings)  # type: ignore
        elif hasattr(self, "value") and self.value is not None:  # type: ignore
            self.cst_name = "value"
            self.cst = self.value  # type: ignore
        else:
            raise AttributeError("No constant is defined for operator 'Constant'.")

    def _run(self, **overridden_attributes):  # type: ignore
        if self.has_linked_attribute:
            if overridden_attributes is None:
                raise RuntimeError(
                    f"Attributes are empty, cannot retrieve value for {self.cst!r}."
                )
            if self.cst_name not in overridden_attributes:
                raise RuntimeError(
                    f"Cannot find attribute {self.cst_name!r} in {list(overridden_attributes)!r}."
                )
            return (overridden_attributes[self.cst_name],)
        return (self._check(self.cst),)


if onnx_opset_version() >= 12:
    Constant = Constant_12
elif onnx_opset_version() >= 11:
    Constant = Constant_11  # type: ignore
elif onnx_opset_version() >= 9:
    Constant = Constant_9  # type: ignore
else:
    Constant = Constant_1  # type: ignore
