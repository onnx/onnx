from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ExpandDynamicShape(Base):


    @staticmethod
    def export():  # type -> None

        def make_graph(node, input_shape, shape_shape, output_shape):
            graph = onnx.helper.make_graph(
                nodes = [node],
                name = 'Expand',
                inputs = [onnx.helper.make_tensor_value_info('X',
                                                             onnx.TensorProto.FLOAT,
                                                             input_shape),
                          onnx.helper.make_tensor_value_info('shape',
                                                             onnx.TensorProto.INT64,
                                                             shape_shape)],
                outputs = [onnx.helper.make_tensor_value_info('Y',
                                                              onnx.TensorProto.FLOAT,
                                                              output_shape)])
            return graph

        node = onnx.helper.make_node(
            'Expand', ['X', 'shape'], ['Y'], name = 'test')
        input_shape = [1, 3, 1]
        x = np.ones(input_shape)

        #1st testcase
        shape = np.array([3, 1])
        y = x * np.ones(shape)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs = [x, shape], outputs = [y], name =
                "test_expand_shape_model1")

        #2nd testcase
        shape = np.array([1, 3])
        y = x * np.ones(shape)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs = [x, shape], outputs = [y], name =
               "test_expand_shape_model2")

        #3rd testcase
        shape = np.array([3, 1, 3])
        y = x * np.ones(shape)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs = [x, shape], outputs = [y], name =
               "test_expand_shape_model3")

        #4th testcase
        shape = np.array([3, 3, 1, 3])
        y = x * np.ones(shape)
        graph = make_graph(node, input_shape, shape.shape, y.shape)
        model = onnx.helper.make_model(graph, producer_name='backend-test')
        expect(model, inputs = [x, shape], outputs = [y], name =
               "test_expand_shape_model4")
