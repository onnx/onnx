from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def softmaxcrossentropy_2d(x, labels):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
    return np.multiply(labels, np.log(p))


class SoftmaxCrossEntropy(Base):

    @staticmethod
    def export_softmaxcrossentropy_none():
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
        l = softmaxcrossentropy_2d(x, labels)

        # Check results
        expect(node, inputs=[x, labels], outputs=[l], name='test_softmax_cross_entropy_none')


    @staticmethod
    def export_softmaxcrossentropy_none_weights():
        # Define operator attributes.
        reduction = 'none'
        w = np.array([0.9, 0.7, 0.8, 0.9], dtype=np.float32)
        weights = onnx.helper.make_tensor("weights", onnx.TensorProto.FLOAT, [4], w)

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropy',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction,
                                     weights=weights
                                     )

        # Define operator inputs.
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]], dtype=np.float32)
        labels = np.array([[0.0320586, 0.08714432, 0.23688284, 0.64391428], [0.0320586, 0.08714432, 0.23688284, 0.64391428]], dtype=np.float32)

        # Compute SoftmaxCrossEntropy
        l = softmaxcrossentropy_2d(x, labels)
        l = np.multiply(w, l)

        # Check results
        expect(node, inputs=[x, labels], outputs=[l], name='test_softmax_cross_entropy_none_weights')


    @staticmethod
    def export_softmaxcrossentropy_sum():
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
        l = softmaxcrossentropy_2d(x, labels)
        rs = np.sum(l)

        # Check results
        expect(node, inputs=[x, labels], outputs=[rs], name='test_softmax_cross_entropy_sum')


    @staticmethod
    def export_softmaxcrossentropy_mean():
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
        l = softmaxcrossentropy_2d(x, labels)
        rm = np.mean(l)

        # Check results
        expect(node, inputs=[x, labels], outputs=[rm], name='test_softmax_cross_entropy_mean')
