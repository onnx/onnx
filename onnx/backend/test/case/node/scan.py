# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
import onnx.parser
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Scan(Base):
    @staticmethod
    def export_scan_8() -> None:
        # Given an input sequence [x1, ..., xN], sum up its elements using a scan
        # returning the final state (x1+x2+...+xN) as well the scan_output
        # [x1, x1+x2, ..., x1+x2+...+xN]
        # Note: the first input (sequence_lens) is optional and omitted via "".
        node = onnx.parser.parse_node(
            """
            y, z = Scan ("", initial, x) <
                num_scan_inputs = 1,
                body = scan_body (float[2] sum_in, float[2] next)
                    => (float[2] sum_out, float[2] scan_out)
                {
                    sum_out  = Add(sum_in, next)
                    scan_out = Identity(sum_out)
                }
            >
            """
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
        node = onnx.parser.parse_node(
            """
            y, z = Scan (initial, x) <
                num_scan_inputs = 1,
                body = scan_body (float[2] sum_in, float[2] next)
                    => (float[2] sum_out, float[2] scan_out)
                {
                    sum_out  = Add(sum_in, next)
                    scan_out = Identity(sum_out)
                }
            >
            """
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
        node = onnx.parser.parse_node(
            """
            y_sum, y_prod, z = Scan (initial_sum, initial_prod, x) <
                num_scan_inputs = 1,
                body = scan_body (float[2] sum_in, float[2] prod_in, float[2] next)
                    => (float[2] sum_out, float[2] prod_out, float[2] scan_out)
                {
                    sum_out  = Add(sum_in, next)
                    prod_out = Mul(prod_in, next)
                    scan_out = Identity(sum_out)
                }
            >
            """
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
        node = onnx.parser.parse_node(
            """
            y, z = Scan (initial, x) <
                num_scan_inputs = 1,
                body = scan_body (float sum_in, float next)
                    => (float sum_out, float scan_out)
                {
                    sum_out  = Add(sum_in, next)
                    scan_out = Identity(sum_out)
                }
            >
            """
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
