from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect

class LSTM(Base):

    @staticmethod
    def export_defaults():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 4

            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = np.ones((1, 4*hidden_size, input_size)).astype(np.float32)
            R = np.ones((1, 4*hidden_size, hidden_size)).astype(np.float32)

            output = np.array([[ 0.70377535,  0.70377535,  0.70377535,  0.70377535],
                               [ 0.96009213,  0.96009213,  0.96009213,  0.96009213],
                               [ 0.99451119,  0.99451119,  0.99451119,  0.99451119]]).astype(np.float32)
                            
            expect(node, inputs=[input, W, R], outputs=[output], name='test_lstm_defaults')

    @staticmethod
    def export_initial_bias():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 4
            custom_bias = 0.1

            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R', 'B'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            W = np.ones((1, 4*hidden_size, input_size)).astype(np.float32)
            R = np.ones((1, 4*hidden_size, hidden_size)).astype(np.float32)

            # Adding custom bias
            W_B = custom_bias * np.ones((1, 4*hidden_size)).astype(np.float32)
            R_B = np.zeros((1, 4*hidden_size)).astype(np.float32)
            B = np.concatenate((W_B, R_B), axis = 1)

            output = np.array([[ 0.70919698,  0.70919698,  0.70919698,  0.70919698],
                               [ 0.96049958,  0.96049958,  0.96049958,  0.96049958],
                               [ 0.99456745,  0.99456745,  0.99456745,  0.99456745]]).astype(np.float32)
                            
            expect(node, inputs=[input, W, R, B], outputs=[output], name='test_lstm_with_initial_bias')

    @staticmethod
    def export_peepholes():
            input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
            
            input_size = 2
            hidden_size = 4

            node = onnx.helper.make_node(
                'LSTM',
                inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
                outputs=['Y'],
                hidden_size=hidden_size
            )

            # Initializing Inputs 
            W = np.ones((1, 4*hidden_size, input_size)).astype(np.float32)
            R = np.ones((1, 4*hidden_size, hidden_size)).astype(np.float32)
            B = np.zeros((1, 8*hidden_size))
            seq_lens = np.repeat(input.shape[0], input.shape[1])
            init_h = np.zeros((1, input.shape[1], hidden_size))
            init_c = np.zeros((1, input.shape[1], hidden_size))
            P = np.ones((1, 3*hidden_size, )).astype(np.float32)

            output = np.array([[ 0.724828  ,  0.724828  ,  0.724828  ,  0.724828  ],
                               [ 0.96014303,  0.96014303,  0.96014303,  0.96014303],
                               [ 0.99451232,  0.99451232,  0.99451232,  0.99451232]]).astype(np.float32)
                            
            expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[output], name='test_lstm_with_peepholes')
