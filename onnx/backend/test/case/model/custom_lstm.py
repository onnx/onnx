from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class CustomLSTM(Base):

    @staticmethod
    def export():  # type: () -> None
        # Custom LSTM example.
        # Inputs:
        #   input   [sequence_length, batch_size, input_size]
        #   state   [2, batch, hidden_size]
        # Weights:
        #   w_i     [4 * hidden_size, input_size]
        #   w_h     [4 * hidden_size, hidden_size]
        #   b_i     [4 * hidden_size]
        #   b_h     [4 * hidden_size]
        #
        # Graph(input, state, w_i, w_h, b_i, b_h):
        #   Tensor[] ins = SplitToSequence(input, axis=0, keepdims=0)
        #   Tensor max_len = SequenceLength(ins)
        #   Tensor(bool) cond = Constant(1)
        #   Tensor h_i, Tensor c_i = Split(state, axis=0, split=[1, 1])
        #   Tensor[] outs, Tensor h_y, Tensor c_y = Loop(max_len, cond, h_i, c_i)
        #       block0(Tensor(int64) i, Tensor(bool) cond, Tensor h, Tensor c)
        #           Tensor in = SequenceAt(ins, i)
        #           Tensor gates = Gemm()

        loop_body = onnx.helper.make_graph(
            [],
            'loop_body',
            [],
            []
        )

        node = onnx.helper.make_node(
            'Loop',
            inputs=[],
            outputs=[],

        )
