from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import onnx
from ..base import Base
from . import expect


class ReduceL1(Base):

    @staticmethod
    def export():

        data = np.array(
            [[[1,2], [3,4]],[[5,6], [7,8]],[[9,10], [11,12]]], 
            dtype=np.float32)

        node = onnx.helper.make_node(
            'ReduceL1',
            inputs=['data'],
            outputs=['reduced'],
            axes = [2],
            keepdims = 0 
        )

        reduced = np.array([
            [  3.,   7.], 
            [ 11.,  15.], 
            [ 19.,  23.]], 
            dtype=np.float32)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_l1_do_not_keep_dims')

        node = onnx.helper.make_node(
            'ReduceL1',
            inputs=['data'],
            outputs=['reduced'],
            axes = [2],
            keepdims = 1
        )

        reduced = np.array([
            [[  3.], [  7.]], 
            [[ 11.], [ 15.]], 
            [[ 19.], [ 23.]]], 
            dtype=np.float32)

        expect(node, inputs=[data], outputs=[reduced],
               name='test_reduce_l1_keep_dims')
