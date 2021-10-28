# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import shutil
import tempfile
import unittest

import onnx
from onnx import helper, TensorProto

class TestComposeFunctions(unittest.TestCase):
    def test_merge(self):  # type: () -> None
        def create_tensor(name):  # type: ignore
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])
        A0 = create_tensor("A0")
        A1 = create_tensor("A1")
        B00 = create_tensor("B00")
        B10 = create_tensor("B10")
        B20 = create_tensor("B20")

        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B00"], "L0_0")
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B10"], "L0_1")
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B20"], "L0_2")

        g0 = helper.make_graph(
            [L0_0, L0_1, L0_2],
            "test1",
            [A0, A1],
            [B00, B10, B20])

        B01 = create_tensor("B01")
        B11 = create_tensor("B11")
        B21 = create_tensor("B21")
        C0 = create_tensor("C0")
        C1 = create_tensor("C1")
        D0 = create_tensor("D0")

        L1_0 = helper.make_node("Add", ["B01", "B11"], ["C0"], "L1_0")
        L1_1 = helper.make_node("Sub", ["B11", "B21"], ["C1"], "L1_1")
        L2_0 = helper.make_node("Mul", ["C0", "C1"], ["D0"], "L2_0")

        g1 = helper.make_graph(
            [L1_0, L1_1, L2_0],
            "test2",
            [B01, B11, B21],
            [D0])

        # Test 1: Connecting all outputs/inputs
        g3 = onnx.compose.merge(g0, g1, io_map=[("B00", "B01"), ("B10", "B11"), ("B20", "B21")])
        self.assertEqual(g3.input, g0.input)
        self.assertEqual(g3.output, g1.output)
        # Edge names are different
        self.assertEqual([item.name for item in g3.node], [item.name for item in g0.node] + [item.name for item in g1.node])

        m3 = helper.make_model(g3, producer_name='test')
        onnx.checker.check_model(m3)

        # Test 2: Connecting some outputs/inputs
        g4 = onnx.compose.merge(g0, g1, io_map=[("B00", "B01"), ("B10", "B11")])

        # B20 <-> B21 not connected. They should still be present in the intputs and outputs of the combined graph
        self.assertEqual(len(g4.input), 3)
        self.assertEqual(g4.input[0], g0.input[0])  # A0
        self.assertEqual(g4.input[1], g0.input[1])  # A1
        self.assertEqual(g4.input[2], g1.input[2])  # B21

        self.assertEqual(len(g4.output), 2)
        self.assertEqual(g4.output[0], g0.output[2])  # B20
        self.assertEqual(g4.output[1], g1.output[0])  # D0

        m4 = helper.make_model(g4, producer_name='test')
        onnx.checker.check_model(m4)


if __name__ == '__main__':
    unittest.main()
