# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import SequenceProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class StringSplit(Base):
    @staticmethod
    def export_basic() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["result"],
            delimiter=".",
        )

        x = np.array(["abc.com", "def.net"]).astype(object)
        result = [np.array(["abc", "com"]).astype(object), np.array(["def", "net"]).astype(object)]
        expect(node, inputs=[x], outputs=[result], name="test_string_split_basic")

    @staticmethod
    def export_maxsplit() -> None:
        node = onnx.helper.make_node(
            "StringSplit",
            inputs=["x"],
            outputs=["result"],
            maxsplit=2,
        )

        x = np.array([["hello world", "def.net"], ["o n n x", "the quick brown fox"]]).astype(object)
        result = [
            [np.array(["hello", "world"]).astype(object), np.array(["def.net"]).astype(object)],
            [np.array(["o", "n", "n x"]).astype(object), np.array(["the", "quick", "brown fox"]).astype(object)]
            ]

        output_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.helper.np_dtype_to_tensor_dtype(np.dtype("object")), (None,))))]
        expect(node, inputs=[x], outputs=[result], name="test_string_split_maxsplit", output_type_protos=output_type_protos)