from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ConvTransposeModel(Base):

    @staticmethod
    def export_with_kernel():  # type: () -> None

        node = onnx.helper.make_node(
            'ConvTranspose', ['x', 'w'], ['y'],
            name='test',
            strides=[3, 2],
            output_shape=[10, 8],
            kernel_shape=[3, 3],
            output_padding=[1, 1]
        )
        graph = onnx.helper.make_graph(
            nodes=[node],
            name='ConvTranspose',
            inputs=[onnx.helper.make_tensor_value_info(
                'x', onnx.TensorProto.FLOAT, [1, 1, 3, 3]),
                onnx.helper.make_tensor_value_info(
                'w', onnx.TensorProto.FLOAT, [1, 2, 3, 3])],
            outputs=[onnx.helper.make_tensor_value_info(
                'y', onnx.TensorProto.FLOAT, [1, 2, 10, 8])])

        model = onnx.helper.make_model(graph, producer_name='backend-test')

        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        y = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]],

                       [[0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)

        expect(model, inputs=[x, W], outputs=[y], name='test_DeConv_with_attr')
