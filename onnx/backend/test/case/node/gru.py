from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class GRU(Base):

    @staticmethod
    def export_defaults():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

            input_size = 2
            hidden_size = 4

            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = np.ones((1, 3 * hidden_size, input_size)).astype(np.float32)
            R = np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)

            output = np.array([[0.04719133, 0.04719133, 0.04719133, 0.04719133],
                               [0.04791026, 0.04791026, 0.04791026, 0.04791026],
                               [0.04792343, 0.04792343, 0.04792343, 0.04792343]]).astype(np.float32)

            expect(node, inputs=[input, W, R], outputs=[output], name='test_gru_defaults')

    @staticmethod
    def export_initial_bias():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

            input_size = 2
            hidden_size = 4
            custom_bias = 0.1

            node = onnx.helper.make_node(
                'GRU',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = np.ones((1, 3 * hidden_size, input_size)).astype(np.float32)
            R = np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)

            # Adding custom bias
            W_B = custom_bias * np.ones((1, 3 * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, 3 * hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis=1)

            output = np.array([[0.04293266, 0.04293266, 0.04293266, 0.04293266],
                               [0.04359724, 0.04359724, 0.04359724, 0.04359724],
                               [0.04360932, 0.04360932, 0.04360932, 0.04360932]]).astype(np.float32)

            expect(node, inputs=[input, W, R, B], outputs=[output], name='test_gru_with_initial_bias')
