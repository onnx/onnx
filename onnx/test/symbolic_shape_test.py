# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import checker, helper, TensorProto, NodeProto, GraphProto, ValueInfoProto, ModelProto, ONNX_ML, SparseTensorProto
from onnx.helper import make_model, make_node, make_tensor, make_tensor_value_info, make_empty_tensor_value_info, make_opsetid, make_tensor_sequence_value_info
from typing import Sequence, Union, Text, Tuple, List, Any, Optional
import onnx.shape_inference
import unittest
import os
import numpy as np  # type: ignore


class TestSymbolicShape(unittest.TestCase):

    def _assert_valueinfo_shape(self, onnx_model, valueinfo):  # type: (ModelProto, List[ValueInfoProto]) -> None
        nameShape = {}
        for v in valueinfo:
            shape = []
            for dim_i in range(len(v.type.tensor_type.shape.dim)):
                shape.append(v.type.tensor_type.shape.dim[dim_i].dim_value)
            nameShape[v.name] = shape

        for v in onnx_model.graph.value_info:
            if v.name in nameShape:
                for dim_i in range(len(nameShape[v.name])):
                    dim = v.type.tensor_type.shape.dim[dim_i]
                    # -1 means it's a symbolic shape
                    if nameShape[v.name][dim_i] == -1:
                        # symbolic shape must exist
                        assert dim.dim_param, '%s' % (onnx_model)
                    else:
                        assert dim.dim_value == nameShape[v.name][dim_i], '%s' % (onnx_model)

    def _count_unqiue_dim_param_number(self, onnx_model):  # type: (ModelProto) -> int
        symbol_shape_set = set()
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            for dim in v.type.tensor_type.shape.dim:
                if dim.dim_param:
                    symbol_shape_set.add(dim.dim_param)
        return len(symbol_shape_set)

    def _get_shape_from_name(self, onnx_model, name):  # type: (ModelProto, Text) -> List[Any]
        shape = []
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            if v.name == name:
                for dim_i in range(len(v.type.tensor_type.shape.dim)):
                    dim_param = v.type.tensor_type.shape.dim[dim_i].dim_param
                    dim_value = v.type.tensor_type.shape.dim[dim_i].dim_value
                    if dim_param:
                        shape.append(dim_param)
                    else:
                        shape.append(dim_value)
        return shape

    def test_clip_enable_symbolic(self):  # type: () -> None
        concat = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['C'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[concat, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'A']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'M'])]
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(inferred_model, [make_tensor_value_info("C", TensorProto.FLOAT, (2, -1))])
        # the symbolic shape of C and output should be the same
        assert self._get_shape_from_name(inferred_model, 'C') == self._get_shape_from_name(inferred_model, 'output')

    def test_two_symbolic_clip(self):  # type: () -> None
        concat1 = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        concat2 = helper.make_node('Concat', inputs=['C', 'D'], outputs=['E'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['E'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[concat1, concat2, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'A']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info('D', TensorProto.FLOAT, [2, 'D'])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'M'])]
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(inferred_model, [
            make_tensor_value_info("C", TensorProto.FLOAT, (2, -1)),
            make_tensor_value_info("E", TensorProto.FLOAT, (2, -1))])
        # the symbolic shape of E and output should be the same
        assert self._get_shape_from_name(inferred_model, 'E') == self._get_shape_from_name(inferred_model, 'output')

    def test_duplicate_symbolic_shape(self):  # type: () -> None
        concat1 = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        concat2 = helper.make_node('Concat', inputs=['C', 'D'], outputs=['E'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['E'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[concat1, concat2, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'unk__0']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info('D', TensorProto.FLOAT, [2, 'D'])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'unk__1'])]
        )

        onnx_model = make_model(graph_def)
        original_count = self._count_unqiue_dim_param_number(onnx_model)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        inferred_count = self._count_unqiue_dim_param_number(inferred_model)
        # new symbol 'unk__2' should be generated to prevent duplicate so the count will be count + 1
        assert inferred_count == original_count + 1, '%s%s' % (inferred_model, onnx_model)


if __name__ == '__main__':
    unittest.main()
