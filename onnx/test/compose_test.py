# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from typing import Text, List

import onnx
import onnx.version_converter
from onnx import helper, parser, checker, ModelProto, GraphProto, ValueInfoProto


def _load_model(m_def):  # type: (Text) -> ModelProto
    '''
    Parses a model from a string representation, including checking the model for correctness
    '''
    m = parser.parse_model(m_def)
    checker.check_model(m)
    return m


def _prefixed(prefix, s):  # type: (Text, Text) -> Text
    '''
    Prefixes a string (if not empty)
    '''
    return prefix + s if len(s) > 0 else s


def _get_shape(value_info):  # type: (ValueInfoProto) -> List[int]
    '''
    Returns a list of integers representing the shape of the provided ValueInfoProto
    '''
    return [value_info.type.tensor_type.shape.dim[d].dim_value
            for d in range(len(value_info.type.tensor_type.shape.dim))]


m1_def = '''
    <
        ir_version: 7,
        opset_import: [ "": 10, "com.microsoft": 1]
    >
    agraph (float[N, M] A0, float[N, M] A1) => (float[N, M] B00, float[N, M] B10, float[N, M] B20)
    {
        B00 = Add(A0, A1)
        B10 = Sub(A0, A1)
        B20 = Mul(A0, A1)
    }
    '''

m2_def = '''
    <
        ir_version: 7,
        opset_import: [ "": 10, "com.microsoft": 1]
    >
    agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21) => (float[N, M] D0)
    {
        C0 = Add(B01, B11)
        C1 = Sub(B11, B21)
        M1 = Mul(C0, C1)
    }
    '''


class TestComposeFunctions(unittest.TestCase):
    def test_case_connect_all_no_name_collision(self):  # type: () -> None
        '''
        Tests a simple scenario where two models without overlapping names are merged by
        connecting all the outputs in the first models to all the inputs in the second model
        '''
        m1, m2 = _load_model(m1_def), _load_model(m2_def)

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        g3 = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map=io_map)
        checker.check_graph(g3)

        def check_graph(g1, g2, g3):  # type: (GraphProto, GraphProto, GraphProto) -> None
            self.assertEqual(g3.input, g1.input)
            self.assertEqual(g3.output, g2.output)
            # Edge names are different
            self.assertEqual([item.name for item in g3.node],
                             [item.name for item in g1.node] + [item.name for item in g2.node])

        check_graph(m1.graph, m2.graph, g3)

        m3 = onnx.compose.merge_models(m1, m2, io_map=io_map)
        checker.check_model(m3)
        check_graph(m1.graph, m2.graph, m3.graph)

    def test_case_connect_partially_no_name_collision(self):  # type: () -> None
        '''
        Tests a scenario where two models without overlapping names are merged by
        connecting some outputs from the first model to some inputs in the second.
        The remaining inputs/outputs should be present in the combined model
        '''
        m1, m2 = _load_model(m1_def), _load_model(m2_def)

        io_map = [("B00", "B01"), ("B10", "B11")]
        g3 = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map=io_map)
        checker.check_graph(g3)

        def check_graph(g1, g2, g4):  # type: (GraphProto, GraphProto, GraphProto) -> None
            # B20 <-> B21 not connected. They should still be present in the intputs and outputs of the combined graph
            self.assertEqual(len(g4.input), 3)
            self.assertEqual(g4.input[0], g1.input[0])  # A0
            self.assertEqual(g4.input[1], g1.input[1])  # A1
            self.assertEqual(g4.input[2], g2.input[2])  # B21

            self.assertEqual(len(g4.output), 2)
            self.assertEqual(g4.output[0], g1.output[2])  # B20
            self.assertEqual(g4.output[1], g2.output[0])  # D0

        check_graph(m1.graph, m2.graph, g3)

        m3 = onnx.compose.merge_models(m1, m2, io_map=io_map)
        checker.check_model(m3)
        check_graph(m1.graph, m2.graph, m3.graph)

    def test_merge_models_with_metadata_props(self):  # type: () -> None
        m1 = _load_model(m1_def)
        helper.set_model_props(m1, {'p1': 'v1', 'p2': 'v2'})

        m2 = _load_model(m2_def)
        helper.set_model_props(m2, {'p3': 'v3', 'p4': 'v4'})

        io_map = [("B00", "B01")]
        m3 = onnx.compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 4

        # Overlap, but same value
        helper.set_model_props(m2, {'p1': 'v1', 'p4': 'v4'})
        m3 = onnx.compose.merge_models(m1, m2, io_map=io_map)
        assert len(m3.metadata_props) == 3

        # Same keys but not same value. Error
        helper.set_model_props(m2, {'p1': 'v5', 'p4': 'v4'})
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2, io_map=io_map)

    def test_error_wrong_input_output_name(self):  # type: () -> None
        '''
        Tests that providing a non existing output/input name in the io_map argument produces an error.
        '''
        m1, m2 = _load_model(m1_def), _load_model(m2_def)

        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2,
                          io_map=[("wrong_outname", "B01"), ("B10", "B11"), ("B20", "B21")])

        # Wrong output name
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2,
                          io_map=[("B00", "wrong_input"), ("B10", "B11"), ("B20", "B21")])

    def test_error_ir_version_mismatch(self):  # type: () -> None
        m1 = _load_model('''
    <
        ir_version: 7,
        opset_import: [ "": 13]
    >
    agraph (float[N, M] X0) => (float[N, M] Y0)
    {
        Y0 = Add(X0, X0)
    }
    ''')

        m2 = _load_model('''
    <
        ir_version: 6,
        opset_import: [ "": 13]
    >
    agraph (float[N, M] X1) => (float[N, M] Y1)
    {
        Y1 = Add(X1, X1)
    }
    ''')
        # Wrong IR version name
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2,
                          io_map=[("Y0", "X1")])

    def test_error_opset_import_mismatch(self):  # type: () -> None
        '''
        Tests that providing models with different operator set imported produces an error
        '''
        m1, m2 = _load_model(m1_def), _load_model(m2_def)
        m1 = helper.make_model(m1.graph, producer_name='test',
                               opset_imports=[onnx.helper.make_opsetid("", 10)])
        m2 = helper.make_model(m2.graph, producer_name='test',
                               opset_imports=[onnx.helper.make_opsetid("", 15)])

        io_map = [("B00", "B01"), ("B10", "B11"), ("B20", "B21")]
        self.assertRaises(ValueError,
                          onnx.compose.merge_models, m1, m2, io_map)

        # Converting to the same Operator set version, should work
        m1 = onnx.version_converter.convert_version(m1, 15)
        m3 = onnx.compose.merge_models(m1, m2, io_map=io_map)
        onnx.checker.check_model(m3)

    def _test_add_prefix(self,
                         rename_nodes=False, rename_edges=False,
                         rename_inputs=False, rename_outputs=False,
                         inplace=False):  # type: (bool, bool, bool, bool, bool) -> None
        m1 = _load_model(m1_def)

        prefix = 'pre/'

        if inplace:
            m2 = ModelProto()
            m2.CopyFrom(m1)
            onnx.compose.add_prefix(m2, prefix,
                                    rename_nodes=rename_nodes,
                                    rename_edges=rename_edges,
                                    rename_inputs=rename_inputs,
                                    rename_outputs=rename_outputs,
                                    inplace=True)
        else:
            m2 = onnx.compose.add_prefix(m1, prefix,
                                        rename_nodes=rename_nodes,
                                        rename_edges=rename_edges,
                                        rename_inputs=rename_inputs,
                                        rename_outputs=rename_outputs)
        g_in = m1.graph
        g_out = m2.graph

        if rename_edges or rename_inputs or rename_outputs:
            name_mapping = {}

            # Rename inputs/outputs/edges. Propagate name changes from and to edges
            if rename_edges:
                for n in g_in.node:
                    for e in n.input:
                        name_mapping[e] = _prefixed(prefix, e)
                    for e in n.output:
                        name_mapping[e] = _prefixed(prefix, e)
            else:
                if rename_inputs:
                    for elem in g_in.input:
                        name_mapping[elem.name] = _prefixed(prefix, elem.name)
                if rename_outputs:
                    for elem in g_in.output:
                        name_mapping[elem.name] = _prefixed(prefix, elem.name)

            for n1, n0 in zip(g_out.node, g_in.node):
                for e1, e0 in zip(n1.input, n0.input):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
                for e1, e0 in zip(n1.output, n0.output):
                    self.assertEqual(name_mapping.get(e0, e0), e1)
            for i1, i0 in zip(g_out.input, g_in.input):
                self.assertEqual(name_mapping.get(i0.name, i0.name), i1.name)
            for o1, o0 in zip(g_out.output, g_in.output):
                self.assertEqual(name_mapping.get(o0.name, o0.name), o1.name)

            if rename_nodes:
                for n1, n0 in zip(g_out.node, g_in.node):
                    self.assertEqual(_prefixed(prefix, n0.name), n1.name)

    def test_add_prefix_nodes(self):  # type: () -> None
        '''
        Tests renaming nodes only
        '''
        self._test_add_prefix(rename_nodes=True)

    def test_add_prefix_edges(self):  # type: () -> None
        '''
        Tests prefixing nodes edges. This will also rename inputs/outputs, since the names are shared
        '''
        self._test_add_prefix(rename_edges=True)

    def test_add_prefix_inputs(self):  # type: () -> None
        '''
        Tests prefixing graph inputs only. Relevant node edges should be renamed as well
        '''
        self._test_add_prefix(rename_inputs=True)

    def test_add_prefix_outputs(self):  # type: () -> None
        '''
        Tests prefixing graph outputs only. Relevant node edges should be renamed as well
        '''
        self._test_add_prefix(rename_outputs=True)

    def test_add_prefix_all(self):  # type: () -> None
        '''
        Tests prefixing all names in the graph
        '''
        self._test_add_prefix(rename_nodes=True, rename_edges=True,
                              rename_inputs=True, rename_outputs=True)

    def test_add_prefix_inplace(self):  # type: () -> None
        '''
        Tests prefixing all names in the graph
        '''
        self._test_add_prefix(rename_nodes=True, rename_edges=True,
                              rename_inputs=True, rename_outputs=True, inplace=True)

    def test_expand_out_dim(self):  # type: () -> None
        '''
        Tests expanding output dimensions. The resulting graph should have the same output names,
        but with one more dimension at the specified index.
        '''
        m1 = _load_model(m1_def)

        def _check_model(m1, m2, dim_idx):  # type: (ModelProto, ModelProto, int) -> None
            for out_g2, out_g1 in zip(m2.graph.output, m1.graph.output):
                self.assertEqual(out_g2.name, out_g1.name)
                self.assertEqual(out_g2.type.tensor_type.elem_type,
                                 out_g1.type.tensor_type.elem_type)
                expected_out_shape = _get_shape(out_g1)
                expected_out_shape.insert(dim_idx, 1)
                self.assertEqual(_get_shape(out_g2), expected_out_shape)

        for dim_idx in [0, 2, -1, -3]:
            m2 = onnx.compose.expand_out_dim(m1, dim_idx)
            _check_model(m1, m2, dim_idx)

        # Test inplace
        m2 = ModelProto()
        m2.CopyFrom(m1)
        dim_idx = 0
        onnx.compose.expand_out_dim(m2, dim_idx, inplace=True)
        _check_model(m1, m2, dim_idx)


if __name__ == '__main__':
    unittest.main()
