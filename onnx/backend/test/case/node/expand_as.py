from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ExpandAs(Base):

    @staticmethod
    def export_expandas_dim_changed():  # type: () -> None
        node = onnx.helper.make_node(
            'ExpandAs',
            inputs=['data', 'other'],
            outputs=['expanded'],
        )
        shape = [3, 1]
        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[1.], [2.], [3.]]
        new_shape = [2, 1, 6]
        expanded = data * np.ones(new_shape, dtype=np.float32)
        #print(expanded)
        #[[[1., 1., 1., 1., 1., 1.],
        #  [2., 2., 2., 2., 2., 2.],
        #  [3., 3., 3., 3., 3., 3.]],
        #
        # [[1., 1., 1., 1., 1., 1.],
        #  [2., 2., 2., 2., 2., 2.],
        #  [3., 3., 3., 3., 3., 3.]]]
        new_shape = np.array(new_shape, dtype=np.int64)
        other = np.random.randn(*new_shape)
        expect(node, inputs=[data, other], outputs=[expanded],
               name='test_expand_dim_changed')

    @staticmethod
    def export_expandas_dim_unchanged():  # type: () -> None
        node = onnx.helper.make_node(
            'ExpandAs',
            inputs=['data', 'other'],
            outputs=['expanded'],
        )
        shape = [3, 1]
        new_shape = [3, 4]
        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[1.], [2.], [3.]]
        expanded = np.tile(data, 4)
        #print(expanded)
        #[[1., 1., 1., 1.],
        # [2., 2., 2., 2.],
        # [3., 3., 3., 3.]]
        new_shape = np.array(new_shape, dtype=np.int64)
        other = np.random.randn(*new_shape)
        expect(node, inputs=[data, other], outputs=[expanded],
               name='test_expand_dim_unchanged')
