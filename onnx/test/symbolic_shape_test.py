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

    def test_clip_enable_symbolic(self):  # type: () -> None

        clip = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['C'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[clip, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'A']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'M'])]
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        vis = list(inferred_model.graph.value_info)
        inferred_vis = [make_tensor_value_info("C", TensorProto.FLOAT, (2, 'unk__0'))]
        onnx.save(onnx_model, 't.onnx')
        assert vis == inferred_vis, '\n%s\n%s\n' % (vis, inferred_vis)

    def test_two_symbolic_clip(self):  # type: () -> None

        clip1 = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        clip2 = helper.make_node('Concat', inputs=['C', 'D'], outputs=['E'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['E'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[clip1, clip2, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'A']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info('D', TensorProto.FLOAT, [2, 'D'])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'M'])]
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        vis = list(inferred_model.graph.value_info)
        inferred_vis = [make_tensor_value_info("C", TensorProto.FLOAT, (2, 'unk__0')),
                    make_tensor_value_info("E", TensorProto.FLOAT, (2, 'unk__1'))]

        assert vis == inferred_vis, '\n%s\n%s\n' % (vis, inferred_vis)

    def test_duplicate_symbolic_shape(self):  # type: () -> None

        clip = helper.make_node('Concat', inputs=['A', 'B'], outputs=['C'], name='Concat', axis=1)
        cast = onnx.helper.make_node('Cast',
            inputs=['C'],
            outputs=['output'],
            to=getattr(TensorProto, 'FLOAT'))
        graph_def = helper.make_graph(name='test_graph',
            nodes=[clip, cast],
            inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 'unk__0']),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 3])],
            outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 'unk__1'])]
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        vis = list(inferred_model.graph.value_info)
        # unk__0 and unk__1 have been used so it will use unk__2
        inferred_vis = [make_tensor_value_info("C", TensorProto.FLOAT, (2, 'unk__2'))]

        assert vis == inferred_vis, '\n%s\n%s\n' % (vis, inferred_vis)


if __name__ == '__main__':
    unittest.main()
