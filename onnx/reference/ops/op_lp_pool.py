# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.ops.op_pool_common import CommonPool


class LpPool(CommonPool):
    def _run(
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        dilations=None,
        kernel_shape=None,
        p=2,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        # float16 and bfloat16 are computed in float32: raising the input to
        # the power p overflows float16 (max 65504) for values above 256
        # when p is 2.
        xc = x.astype(np.float32) if x.dtype in (np.float16, ml_dtypes.bfloat16) else x
        # utilize AvgPool the same fashion Pytorch does. Note that there is a difference in computation.
        # it needs another PR to address.
        # https://github.com/pytorch/pytorch/blob/f58ba553b78db7f88477f9ba8c9333bd1590e30a/torch/nn/functional.py#L1015
        power_average = CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            np.power(np.absolute(xc), p),
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )

        kernel_element_count = np.prod(kernel_shape, dtype=np.int64)
        return (
            np.power(kernel_element_count * power_average[0], 1.0 / p).astype(x.dtype),
        )
