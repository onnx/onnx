# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx import TensorProto, helper, numpy_helper
from onnx.reference.custom_element_types import (
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun


def dynamic_quantize_linear_int(x, dtype):
    if dtype == TensorProto.UINT8:
        dtype, qmin, qmax = np.uint8, 0, 255
    else:
        dtype, qmin, qmax = np.int8, -127, 127
    maxx = np.float32(np.maximum(0, np.max(x)))
    minx = np.float32(np.minimum(0, np.min(x)))
    y_scale = np.float32(1.0 if maxx == minx else (maxx - minx)) / np.float32(
        qmax - qmin
    )

    # scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);

    initial_zero_point = np.float32(qmin) - minx / y_scale
    zp = max(qmin, min(qmax, initial_zero_point))
    zpi = np.rint(zp)

    y = np.clip(np.rint(x / y_scale) + zpi, qmin, qmax)
    return (y.astype(dtype), y_scale.astype(x.dtype), zpi.astype(dtype))


class DynamicQuantizeLinear_11(OpRun):
    def _run(self, x):  # type: ignore
        # args: x, y_scale, zero_point
        return dynamic_quantize_linear_int(x, TensorProto.UINT8)


def estimation_quantization_scale(
    coef: np.array, to: int = TensorProto.FLOAT8E4M3FN
) -> tuple[float, float]:
    """
    Estimates the scale parameter for the quantization to float 8 assuming
    the distribution of the coefficients is gaussian.
    """

    if to == TensorProto.FLOAT8E4M3FN:
        fct = numpy_helper.float8e4m3_to_float32
    elif to == TensorProto.FLOAT8E4M3FNUZ:

        def fct(x):
            return numpy_helper.float8e4m3_to_float32(x, uz=True)

    elif to == TensorProto.FLOAT8E5M2:
        fct = numpy_helper.float8e5m2_to_float32
    elif to == TensorProto.FLOAT8E5M2FNUZ:

        def fct(x):
            return numpy_helper.float8e5m2_to_float32(x, uz=True, fn=True)

    else:
        raise ValueError(f"Unexpected to={to!r}.")

    float8 = [fct(i) for i in range(0, 256)]
    quant_float = [f for f in float8 if not np.isnan(f)]
    cr = coef.ravel()
    ca = np.abs(cr)
    ca_den = ca.copy()
    ca_den[ca == 0] = 1
    std_coef = np.std(ca ** (1.0 / 3.0) * cr / ca_den)
    std_quant = np.std(np.array(quant_float, dtype=np.float32))
    scale = std_quant / std_coef

    return 1.0 / scale


class DynamicQuantizeLinear_20(OpRun):
    def _run(self, x, to=None):  # type: ignore
        # args: x, y_scale, zero_point
        if to is None or to == TensorProto.UINT8:
            return dynamic_quantize_linear_int(x, TensorProto.UINT8)
        if to == TensorProto.INT8:
            return dynamic_quantize_linear_int(x, TensorProto.INT8)

        y_scale = estimation_quantization_scale(x, to)
        y = x / y_scale

        if to == TensorProto.FLOAT8E4M3FN:
            fct = helper.float32_to_float8e4m3
            dtype = float8e4m3fn
        elif to == TensorProto.FLOAT8E4M3FNUZ:

            def fct(x):
                return helper.float32_to_float8e4m3(x, uz=True)

            dtype = float8e4m3fnuz
        elif to == TensorProto.FLOAT8E5M2:
            fct = helper.float32_to_float8e5m2
            dtype = float8e5m2
        elif to == TensorProto.FLOAT8E5M2FNUZ:

            def fct(x):
                return helper.float32_to_float8e5m2s(x, uz=True, fn=True)

            dtype = float8e5m2fnuz
        else:
            raise ValueError(f"Unexpected to={to!r}.")

        fv = np.vectorize(fct)

        def cvt(x):
            return fv(x).astype(dtype)

        return (cvt(y), y_scale.astype(x.dtype), cvt(np.array([0], dtype=np.float32)))
