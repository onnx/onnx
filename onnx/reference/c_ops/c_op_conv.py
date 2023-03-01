# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,R0915,R1702,W0221

from typing import Any, Dict

import numpy as np

from onnx import NodeProto
from onnx.onnx_cpp2py_export.c_ops import ConvFloat, ConvDouble
from onnx.reference.op_run import OpRun


class Conv(OpRun):
    def __init__(
        self, onnx_node: NodeProto, run_params: Dict[str, Any], schema: Any = None
    ):
        OpRun.__init__(self, onnx_node, run_params, schema)
        self.cache_ = {}

    def _run(  # type: ignore
        self,
        X,
        W,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        if X.dtype not in self.cache_:
            if X.dtype == np.float32:
                rt = ConvFloat()
            elif X.dtype == np.float64:
                rt = ConvDouble()
            else:
                raise TypeError(
                    f"No C implementation C for operator 'Conv' and dtype={X.dtype}."
                )
            self.cache_[X.dtype] = rt

            rt.init(
                auto_pad,
                np.array(dilations or [], dtype=np.int64),
                group,
                np.array(kernel_shape or [], dtype=np.int64),
                np.array(pads or [], dtype=np.int64),
                np.array(strides or [], dtype=np.int64),
            )

        rt = self.cache_[X.dtype]

        if X is None:
            raise ValueError(  # pragma: no cover
                "X cannot be None for operator %r, ONNX=%r"
                % (type(self), self.onnx_node)
            )
        if min(X.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. X.shape={X.shape!r}."
            )
        if min(W.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. W.shape={W.shape!r}."
            )
        if B is not None and min(B.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Unable to run operator Conv on an empty matrix. B.shape={B.shape!r}."
            )
        cv = rt.compute(X, W, B)
        return (cv,)
