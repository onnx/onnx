# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from typing import Text, List

import onnx
import onnx.version_converter
from onnx import helper, TensorProto, GraphProto, ValueInfoProto


def _create_tensor(name, dtype=TensorProto.FLOAT, shape=[1, 2]):  # type: ignore
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


class TestComposeFunctions(unittest.TestCase):
    def test_merge(self):  # type: () -> None
        A0 = _create_tensor("A0")
        A1 = _create_tensor("A1")
        B00 = _create_tensor("B00")
        B10 = _create_tensor("B10")
        B20 = _create_tensor("B20")

        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B00"], "L0_0")
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B10"], "L0_1")
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B20"], "L0_2")

        g1 = helper.make_graph(
            [L0_0, L0_1, L0_2],
            "test1",
            [A0, A1],
            [B00, B10, B20])

        B01 = _create_tensor("B01")
        B11 = _create_tensor("B11")
        B21 = _create_tensor("B21")
        D0 = _create_tensor("D0")

        L1_0 = helper.make_node("Add", ["B01", "B11"], ["C0"], "L1_0")
        L1_1 = helper.make_node("Sub", ["B11", "B21"], ["C1"], "L1_1")
        L2_0 = helper.make_node("Mul", ["C0", "C1"], ["D0"], "L2_0")

        g2 = helper.make_graph(
            [L1_0, L1_1, L2_0],
            "test2",
            [B01, B11, B21],
            [D0])

        # Test 1: Connecting all outputs/inputs
        io_map_g3 = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        g3 = onnx.compose.merge_graphs(
            g1, g2, io_map=io_map_g3)

        def check_g3(g1, g2, g3):  # type: (GraphProto, GraphProto, GraphProto) -> None
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            # Edge names are different
            self.assertEqual([item.name for item in g3.node],
                             [item.name for item in g1.node] + [item.name for item in g2.node])

        check_g3(g1, g2, g3)
        m3 = helper.make_model(g3, producer_name='test',
                               opset_imports=[onnx.helper.make_opsetid("", 15)])
        onnx.checker.check_model(m3)

        # Test merge models API
        m1 = helper.make_model(g1, producer_name='test',
                               opset_imports=[onnx.helper.make_opsetid("", 15)])
        m2 = helper.make_model(g2, producer_name='test',
                               opset_imports=[onnx.helper.make_opsetid("", 15)])

        m3 = onnx.compose.merge_models(m1, m2, io_map_g3)
        onnx.checker.check_model(m3)
        check_g3(m1.graph, m2.graph, m3.graph)

        # Test 2: Connecting some outputs/inputs
        io_map_g4 = [("B00", "B01"), ("B10", "B11")]
        g4 = onnx.compose.merge_graphs(g1, g2, io_map=io_map_g4)

        def check_g4(g1, g2, g4):  # type: (GraphProto, GraphProto, GraphProto) -> None
            # B20 <-> B21 not connected. They should still be present in the intputs and outputs of the combined graph
            self.assertEqual(len(g4.input), 3)
            self.assertEqual(g4.input[0], g1.input[0])  # A0
            self.assertEqual(g4.input[1], g1.input[1])  # A1
            self.assertEqual(g4.input[2], g2.input[2])  # B21

            self.assertEqual(len(g4.output), 2)
            self.assertEqual(g4.output[0], g1.output[2])  # B20
            self.assertEqual(g4.output[1], g2.output[0])  # D0

        check_g4(g1, g2, g4)
        m4 = helper.make_model(g4, producer_name='test')
        onnx.checker.check_model(m4)

        # Test merge models API
        m4 = onnx.compose.merge_models(m1, m2, io_map_g4)
        onnx.checker.check_model(m4)
        check_g4(m1.graph, m2.graph, m4.graph)

        # Wrong output name
        self.assertRaises(ValueError,
                          onnx.compose.merge_graphs, g1, g2, io_map=[("wrong_outname", "B01"), ("B10", "B11"), ("B20", "B21")])

        # Wrong output name
        self.assertRaises(ValueError,
                          onnx.compose.merge_graphs, g1, g2, io_map=[("B00", "wrong_input"), ("B10", "B11"), ("B20", "B21")])

        # Wrong IR version.
        min_ir_version = helper.find_min_ir_version_for([entry for entry in m1.opset_import])
        wrong_ir_version = min_ir_version - 1
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2, io_map=io_map_g3, ir_version=wrong_ir_version)

        # Minimum IR version should work
        m3 = onnx.compose.merge_models(
            m1, m2, io_map=io_map_g3, ir_version=min_ir_version)
        onnx.checker.check_model(m3)
        check_g3(m1.graph, m2.graph, m3.graph)

        # Not compatible operator sets
        m1_10 = helper.make_model(g1, producer_name='test',
                                  opset_imports=[onnx.helper.make_opsetid("", 10)])
        m2_15 = helper.make_model(g2, producer_name='test',
                                  opset_imports=[onnx.helper.make_opsetid("", 15)])
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1_10, m2_15, io_map=io_map_g3)

        # Converting to the same Operator set version, should work
        m1_15 = onnx.version_converter.convert_version(m1_10, 15)
        m3 = onnx.compose.merge_models(m1_15, m2_15, io_map=io_map_g3)
        onnx.checker.check_model(m3)
        check_g3(m1.graph, m2.graph, m3.graph)

    def test_add_prefix(self):  # type: () -> None
        A0 = _create_tensor("A0")
        A1 = _create_tensor("A1")
        B00 = _create_tensor("B00")
        B10 = _create_tensor("B10")
        B20 = _create_tensor("B20")

        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B00"], "L0_0")
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B10"], "L0_1")
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B20"], "L0_2")

        g0 = helper.make_graph(
            [L0_0, L0_1, L0_2],
            "test1",
            [A0, A1],
            [B00, B10, B20])

        g1 = onnx.compose.add_prefix(g0, 'g0/')

        def prefixed(prefix, s):  # type: (Text, Text) -> Text
            return prefix + s if len(s) > 0 else s

        for in1, in0 in zip(g1.input, g0.input):
            self.assertEqual(in1.name, prefixed('g0/', in0.name))

        for out1, out0 in zip(g1.output, g0.output):
            self.assertEqual(out1.name, prefixed('g0/', out0.name))

        for n1, n0 in zip(g1.node, g0.node):
            self.assertEqual(n1.name, prefixed('g0/', n0.name))

            for e1, e0 in zip(n1.input, n0.input):
                self.assertEqual(e1, prefixed('g0/', e0))

            for e1, e0 in zip(n1.output, n0.output):
                self.assertEqual(e1, prefixed('g0/', e0))

        m = helper.make_model(g1, producer_name='test')
        onnx.checker.check_model(m)

    def test_expand_out_dim(self):  # type: () -> None
        A0 = _create_tensor("A0", shape=[5, 4])
        A1 = _create_tensor("A1", shape=[5, 4])
        B0 = _create_tensor("B0", shape=[3, 2])
        B1 = _create_tensor("B1", shape=[3, 2])
        C0 = _create_tensor("C0", shape=[5, 4])
        C1 = _create_tensor("C1", shape=[3, 2])

        L0 = helper.make_node("Add", ["A0", "A1"], ["C0"], "L0")
        L1 = helper.make_node("Sub", ["B0", "B1"], ["C1"], "L1")

        g0 = helper.make_graph(
            [L0, L1],
            "test1",
            [A0, A1, B0, B1],
            [C0, C1])

        def out_shape(out):  # type: (ValueInfoProto) -> List[int]
            return [out.type.tensor_type.shape.dim[d].dim_value
                    for d in range(len(out.type.tensor_type.shape.dim))]

        for dim_idx in [0, 2, -1, -3]:
            g1 = onnx.compose.expand_out_dim(g0, dim_idx)

            for out0, out1 in zip(g0.output, g1.output):
                self.assertEqual(out1.name, out0.name + '_expanded')
                self.assertEqual(out1.type.tensor_type.elem_type,
                                 out0.type.tensor_type.elem_type)
                expected_out_shape = out_shape(out0)
                expected_out_shape.insert(dim_idx, 1)
                self.assertEqual(out_shape(out1), expected_out_shape)

            m = helper.make_model(g1, producer_name='test')
            onnx.checker.check_model(m)


if __name__ == '__main__':
    unittest.main()
