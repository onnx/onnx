# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Scan(Base):
    @staticmethod
    def export_scan_8() -> None:
        # Given an input sequence [x1, ..., xN], sum up its elements using a scan
        # returning the final state (x1+x2+...+xN) as well the scan_output
        # [x1, x1+x2, ..., x1+x2+...+xN]
        sum_in = onnx.helper.make_tensor_value_info(
            "sum_in", onnx.TensorProto.FLOAT, [2]
        )
        next_ = onnx.helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info(
            "sum_out", onnx.TensorProto.FLOAT, [2]
        )
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, [2]
        )
        add_node = onnx.helper.make_node(
            "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
        )
        id_node = onnx.helper.make_node(
            "Identity", inputs=["sum_out"], outputs=["scan_out"]
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node], "scan_body", [sum_in, next_], [sum_out, scan_out]
        )
        # create scan op node
        no_sequence_lens = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "Scan",
            inputs=[no_sequence_lens, "initial", "x"],
            outputs=["y", "z"],
            num_scan_inputs=1,
            body=scan_body,
        )
        # create inputs for batch-size 1, sequence-length 3, inner dimension 2
        initial = np.array([0, 0]).astype(np.float32).reshape((1, 2))
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((1, 3, 2))
        # final state computed = [1 + 3 + 5, 2 + 4 + 6]
        y = np.array([9, 12]).astype(np.float32).reshape((1, 2))
        # scan-output computed
        z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((1, 3, 2))

        expect(
            node,
            inputs=[initial, x],
            outputs=[y, z],
            name="test_scan_sum",
            opset_imports=[onnx.helper.make_opsetid("", 8)],
        )

    @staticmethod
    def export_scan_9() -> None:
        # Given an input sequence [x1, ..., xN], sum up its elements using a scan
        # returning the final state (x1+x2+...+xN) as well the scan_output
        # [x1, x1+x2, ..., x1+x2+...+xN]
        sum_in = onnx.helper.make_tensor_value_info(
            "sum_in", onnx.TensorProto.FLOAT, [2]
        )
        next_ = onnx.helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info(
            "sum_out", onnx.TensorProto.FLOAT, [2]
        )
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, [2]
        )
        add_node = onnx.helper.make_node(
            "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
        )
        id_node = onnx.helper.make_node(
            "Identity", inputs=["sum_out"], outputs=["scan_out"]
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node], "scan_body", [sum_in, next_], [sum_out, scan_out]
        )
        # create scan op node
        node = onnx.helper.make_node(
            "Scan",
            inputs=["initial", "x"],
            outputs=["y", "z"],
            num_scan_inputs=1,
            body=scan_body,
        )
        # create inputs for sequence-length 3, inner dimension 2
        initial = np.array([0, 0]).astype(np.float32).reshape((2,))
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))
        # final state computed = [1 + 3 + 5, 2 + 4 + 6]
        y = np.array([9, 12]).astype(np.float32).reshape((2,))
        # scan-output computed
        z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))

        expect(
            node,
            inputs=[initial, x],
            outputs=[y, z],
            name="test_scan9_sum",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )

    @staticmethod
    def export_scan_9_multi_state() -> None:
        # Scan with two state variables: running sum and running product.
        # This exercises the case where num_loop_state_vars (2) differs from
        # num_scan_inputs (1).
        #
        # Body inputs:  sum_in (state), prod_in (state), next (scan)
        # Body outputs: sum_out (state), prod_out (state), scan_out (scan)
        sum_in = onnx.helper.make_tensor_value_info(
            "sum_in", onnx.TensorProto.FLOAT, [2]
        )
        prod_in = onnx.helper.make_tensor_value_info(
            "prod_in", onnx.TensorProto.FLOAT, [2]
        )
        next_ = onnx.helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, [2])
        sum_out = onnx.helper.make_tensor_value_info(
            "sum_out", onnx.TensorProto.FLOAT, [2]
        )
        prod_out = onnx.helper.make_tensor_value_info(
            "prod_out", onnx.TensorProto.FLOAT, [2]
        )
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, [2]
        )
        add_node = onnx.helper.make_node(
            "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
        )
        mul_node = onnx.helper.make_node(
            "Mul", inputs=["prod_in", "next"], outputs=["prod_out"]
        )
        id_node = onnx.helper.make_node(
            "Identity", inputs=["sum_out"], outputs=["scan_out"]
        )
        scan_body = onnx.helper.make_graph(
            [add_node, mul_node, id_node],
            "scan_body",
            [sum_in, prod_in, next_],
            [sum_out, prod_out, scan_out],
        )
        node = onnx.helper.make_node(
            "Scan",
            inputs=["initial_sum", "initial_prod", "x"],
            outputs=["y_sum", "y_prod", "z"],
            num_scan_inputs=1,
            body=scan_body,
        )
        # x = [[1, 2], [3, 4], [5, 6]]
        initial_sum = np.array([0, 0]).astype(np.float32)
        initial_prod = np.array([1, 1]).astype(np.float32)
        x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))
        # final sum = [1+3+5, 2+4+6] = [9, 12]
        y_sum = np.array([9, 12]).astype(np.float32)
        # final product = [1*3*5, 2*4*6] = [15, 48]
        y_prod = np.array([15, 48]).astype(np.float32)
        # scan output (running sum) = [[1,2], [4,6], [9,12]]
        z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))

        expect(
            node,
            inputs=[initial_sum, initial_prod, x],
            outputs=[y_sum, y_prod, z],
            name="test_scan9_multi_state",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )

    @staticmethod
    def export_scan_9_scalar() -> None:
        # Scan with scalar state and scan output to verify that output
        # shapes are not distorted (e.g. (T,) not (T, 1)).
        sum_in = onnx.helper.make_tensor_value_info(
            "sum_in", onnx.TensorProto.FLOAT, []
        )
        next_ = onnx.helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, [])
        sum_out = onnx.helper.make_tensor_value_info(
            "sum_out", onnx.TensorProto.FLOAT, []
        )
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, []
        )
        add_node = onnx.helper.make_node(
            "Add", inputs=["sum_in", "next"], outputs=["sum_out"]
        )
        id_node = onnx.helper.make_node(
            "Identity", inputs=["sum_out"], outputs=["scan_out"]
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node], "scan_body", [sum_in, next_], [sum_out, scan_out]
        )
        node = onnx.helper.make_node(
            "Scan",
            inputs=["initial", "x"],
            outputs=["y", "z"],
            num_scan_inputs=1,
            body=scan_body,
        )
        initial = np.float32(0.0)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        # final state = 1+2+3+4+5 = 15
        y = np.float32(15.0)
        # scan output = [1, 3, 6, 10, 15], shape (5,)
        z = np.array([1, 3, 6, 10, 15]).astype(np.float32)

        expect(
            node,
            inputs=[initial, x],
            outputs=[y, z],
            name="test_scan9_scalar",
            opset_imports=[onnx.helper.make_opsetid("", 9)],
        )
