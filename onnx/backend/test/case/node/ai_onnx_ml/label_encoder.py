# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor


class LabelEncoder(Base):
    @staticmethod
    def export_string_int_label_encoder() -> None:
        node = onnx.helper.make_node(
            "LabelEncoder",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            keys_strings=["a", "b", "c"],
            values_int64s=[0, 1, 2],
            default_int64=42,
        )
        x = np.array(["a", "b", "d", "c", "g"]).astype(object)
        y = np.array([0, 1, 42, 2, 42]).astype(np.int64)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_label_encoder_string_int",
        )

        node = onnx.helper.make_node(
            "LabelEncoder",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            keys_strings=["a", "b", "c"],
            values_int64s=[0, 1, 2],
        )
        x = np.array(["a", "b", "d", "c", "g"]).astype(object)
        y = np.array([0, 1, -1, 2, -1]).astype(np.int64)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_label_encoder_string_int_no_default",
        )

    @staticmethod
    def export_tensor_based_label_encoder() -> None:
        node = onnx.helper.make_node(
            "LabelEncoder",
            inputs=["X"],
            outputs=["Y"],
            domain="ai.onnx.ml",
            keys_as_tensor=make_tensor(
                "keys_as_tensor", onnx.TensorProto.STRING, (3,), ["a", "b", "c"]
            ),
            values_as_tensor=make_tensor(
                "values_as_tensor", onnx.TensorProto.INT16, (3,), [0, 1, 2]
            ),
            default_as_tensor=make_tensor(
                "default_as_tensor", onnx.TensorProto.INT16, (1,), [42]
            ),
        )
        x = np.array(["a", "b", "d", "c", "g"]).astype(object)
        y = np.array([0, 1, 42, 2, 42]).astype(np.int16)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_ai_onnx_ml_label_encoder_tensor_mapping",
        )
