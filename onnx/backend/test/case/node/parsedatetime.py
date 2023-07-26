# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime

import numpy as np

import onnx
from onnx import numpy_helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class ParseDateTime(Base):
    @staticmethod
    def export_float_nan_default() -> None:
        fmt = "%d/%m/%y %H:%M"
        default = float("NaN")
        node = onnx.helper.make_node(
            "ParseDateTime",
            inputs=["x"],
            outputs=["y"],
            format=fmt,
            unit="s",
            default=onnx.helper.make_tensor(
                name="default",
                data_type=onnx.TensorProto.DOUBLE,
                dims=[],
                vals=np.array(default),
            ),
        )
        x = np.array(["21/11/06 16:30", "foobar"], dtype=object)
        y = []
        for s in x:
            try:
                # datetime.timestamp() returns a float
                y.append(datetime.strptime(s, fmt).timestamp())
            except ValueError:
                y.append(default)
        expect(node, inputs=[x], outputs=[np.array(y)], name="test_parsedatetime")

    @staticmethod
    def export_int_default() -> None:
        fmt = "%d/%m/%y %H:%M"
        default = np.iinfo(np.int64).min
        node = onnx.helper.make_node(
            "ParseDateTime",
            inputs=["x"],
            outputs=["y"],
            format=fmt,
            unit="s",
            default=onnx.helper.make_tensor(
                name="default",
                data_type=onnx.TensorProto.INT64,
                dims=[],
                vals=np.array(default),
            ),
        )
        x = np.array(["21/11/06 16:30", "foobar"], dtype=object)
        y = []
        for s in x:
            try:
                y.append(datetime.strptime(s, fmt).timestamp())
            except ValueError:
                y.append(default)
        expect(
            node, inputs=[x], outputs=[np.array(y, np.int64)], name="test_parsedatetime"
        )
