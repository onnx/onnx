# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class StringConcat(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "StringConcat",
            inputs=["x", "y"],
            outputs=["result"],
        )
        x = np.array(["abc", "def"]).astype("object")
        y = np.array([".com", ".net"]).astype("object")
        result = np.array(["abc.com", "def.net"]).astype("object")

        expect(node, inputs=[x, y], outputs=[result], name="test_string_concat")

        x = np.array(["cat", "dog", "snake"]).astype("object")
        y = np.array(["s"]).astype("object")
        result = np.array(["cats", "dogs", "snakes"]).astype("object")

        expect(
            node,
            inputs=[x, y],
            outputs=[result],
            name="test_string_concat_broadcasting",
        )

        x = np.array("cat").astype("object")
        y = np.array("s").astype("object")
        result = np.array("cats").astype("object")

        expect(
            node,
            inputs=[x, y],
            outputs=[result],
            name="test_string_concat_zero_dimensional",
        )
