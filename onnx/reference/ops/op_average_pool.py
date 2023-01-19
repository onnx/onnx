# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221,R0913,R0914

from onnx.defs import onnx_opset_version
from ._op_common_pool import CommonPool


class AveragePool_1(CommonPool):
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


class AveragePool_7(CommonPool):
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


class AveragePool_11(CommonPool):
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


class AveragePool_19(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        dilations=None,
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
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )


if onnx_opset_version() > 11:
    AveragePool = AveragePool_19
elif onnx_opset_version() > 7:
    AveragePool = AveragePool_11  # type: ignore
elif onnx_opset_version() > 1:
    AveragePool = AveragePool_7  # type: ignore
else:
    AveragePool = AveragePool_1  # type: ignore
