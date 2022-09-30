# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221,R0913,R0914

from ._op_common_pool import CommonPool


class AveragePool(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        return CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=None,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
