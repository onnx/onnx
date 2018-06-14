from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Expand(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Expand',
            inputs=['data', 'new_shape'],
            outputs=['expanded'],
        )
        shape = [3, 1]
        new_shape = [3, 4]
        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        #print(data)
        #[[1.], [2.], [3.]]
        expanded = np.reshape(np.array([map(lambda x: np.repeat(x, 4), data)]), new_shape)
        #print(expanded)
        #[[1., 1., 1., 1.],
        # [2., 2., 2., 2.],
        # [3., 3., 3., 3.]]
        new_shape = np.array(new_shape)
        expect(node, inputs=[data, new_shape], outputs=[expanded],
               name='test_expand_dim_unchanged')
        new_shape = [2, 1, 6]
        expanded = data * np.ones(new_shape)
        #print(expanded)
        #[[[1., 1., 1., 1., 1., 1.],
        #  [2., 2., 2., 2., 2., 2.],
        #  [3., 3., 3., 3., 3., 3.]],
        #
        # [[1., 1., 1., 1., 1., 1.],
        #  [2., 2., 2., 2., 2., 2.],
        #  [3., 3., 3., 3., 3., 3.]]]
        new_shape = np.array(new_shape)
        expect(node, inputs=[data, new_shape], outputs=[expanded],
               name='test_expand_dim_changed')
