# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
        result = CommonPool._run(
            self,
            "LPPOOL",
            count_include_pad,
            x.astype(np.float64),
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            p=p,
        )
        return (result[0].astype(x.dtype),)
