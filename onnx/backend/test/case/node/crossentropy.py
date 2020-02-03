from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def softmax_2d(x):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


class SoftmaxCrossEntropy(Base):

    @staticmethod
    def export_crossentropy_none():
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropy',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction
                                     )

        # Define operator inputs.
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]], dtype=np.float32)
        labels = np.array([[0.0320586, 0.08714432, 0.23688284, 0.64391428], [0.0320586, 0.08714432, 0.23688284, 0.64391428]], dtype=np.float32)
        
        # Compute SoftmaxCrossEntropy
        p = softmax_2d(x)
        l = np.multiply(labels, p)

        # Check results
        expect(node, inputs=[x, labels], outputs=[l], name='test_cross_entropy_none')


    @staticmethod
    def export_crossentropy_none_weights():
        # Define operator attributes.
        reduction = 'none'
        weights = [[0.9, 0.8, 0.9, 0.8], [0.9, 0.7, 0.8, 0.9]]

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropy',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction
                                     )

        # Define operator inputs.
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]], dtype=np.float32)
        labels = np.array([[0.0320586, 0.08714432, 0.23688284, 0.64391428], [0.0320586, 0.08714432, 0.23688284, 0.64391428]], dtype=np.float32)

        # Compute SoftmaxCrossEntropy
        p = softmax_2d(x)
        l = np.multiply(labels, p)
        l = np.multiply(weights, l)

        # Check results
        expect(node, inputs=[x, labels], outputs=[l], name='test_cross_entropy_none_weights')


    @staticmethod
    def export_crossentropy_sum():
        # Define operator attributes.
        reduction = 'sum'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropy',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction
                                     )

        # Define operator inputs.
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]], dtype=np.float32)
        labels = np.array([[0.0320586, 0.08714432, 0.23688284, 0.64391428], [0.0320586, 0.08714432, 0.23688284, 0.64391428]], dtype=np.float32)

        # Compute SoftmaxCrossEntropy
        p = softmax_2d(x)
        l = np.multiply(labels, p)
        r = np.sum(l, axis=1)

        # Check results
        expect(node, inputs=[x, labels], outputs=[r], name='test_cross_entropy_sum')


    @staticmethod
    def export_crossentropy_mean():
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropy',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction
                                     )

        # Define operator inputs.
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]], dtype=np.float32)
        labels = np.array([[0.0320586, 0.08714432, 0.23688284, 0.64391428], [0.0320586, 0.08714432, 0.23688284, 0.64391428]], dtype=np.float32)

        # Compute SoftmaxCrossEntropy
        p = softmax_2d(x)
        l = np.multiply(labels, p)
        r = np.mean(l, axis=1)

        # Check results
        expect(node, inputs=[x, labels], outputs=[r], name='test_cross_entropy_mean')
