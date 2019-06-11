from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
import random
import string
import onnx
from ..base import Base
from . import expect

def string_generator(size, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class Equal(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_equal')

    @staticmethod
    def export_equal_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )

        x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
        y = (np.random.randn(5) * 10).astype(np.int32)
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_equal_bcast')

    @staticmethod
    def export_string_equal():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )
        vfunc = np.vectorize(string_generator)
        x = vfunc(np.random.randint(0, 10, size=(3, 4, 5))).astype(np.object)
        y = shuffle_along_axis(x, -1).astype(np.object)  # increase collision probability
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_string_equal')

    @staticmethod
    def export_string_equal_broadcast():  # type: () -> None
        node = onnx.helper.make_node(
            'Equal',
            inputs=['x', 'y'],
            outputs=['z'],
        )
        vfunc = np.vectorize(string_generator)
        x = vfunc(np.random.randint(0, 10, size=(3, 4, 5))).astype(np.object)
        y = np.random.choice(list(set(x.flatten())), 5).astype(np.object) # increase collision probability
        z = np.equal(x, y)
        expect(node, inputs=[x, y], outputs=[z],
               name='test_string_equal_bcast')