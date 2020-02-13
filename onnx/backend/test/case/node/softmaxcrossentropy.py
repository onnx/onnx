from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def softmaxcrossentropy_2d(x, target, weight=None, reduction='mean'):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
    inp = np.log(p)
    input_shape = inp.shape
    N, C = input_shape
    neg_gather_element_input = np.zeros((N, ), dtype=np.float32)
    for i in range(N):
        neg_gather_element_input[i] = -inp[i][target[i]]

    loss = neg_gather_element_input
    if weight is not None:
        gather_weight = np.take(weight, target)
        loss = gather_weight * loss
        if reduction == 'mean':
            return loss.sum() / gather_weight.sum()

    if reduction == 'mean':
        loss = np.mean(loss)
    if reduction == 'sum':
        loss = np.sum(loss)

    return loss

def softmaxcrossentropy_3d(x, target, weight=None, reduction='mean'):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    inp = np.log(p)
    input_shape = inp.shape
    N, C, D = input_shape
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            neg_gather_element_input[i][d] = -inp[i][target[i][d]][d]

    loss = neg_gather_element_input
    if weight is not None:
        gather_weight = np.take(weight, target)
        loss = gather_weight * loss
        if reduction == 'mean':
            return loss.sum() / gather_weight.sum()

    if reduction == 'mean':
        loss = np.mean(loss)
    if reduction == 'sum':
        loss = np.sum(loss)

    return loss


class SoftmaxCrossEntropyLoss(Base):

#    @staticmethod
#    def export_softmaxcrossentropy_none():
#        # Define operator attributes.
#        reduction = 'none'
#
#        # Create operator.
#        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
#                                     inputs=['x', 'y'],
#                                     outputs=['z'],
#                                     reduction=reduction)
#
#        # Define operator inputs.
#        np.random.seed(0)
#        x = np.random.rand(3, 5).astype(np.float32)
#        labels = np.random.randint(0, high=5, size=(3, ))
#
#        # Compute SoftmaxCrossEntropyLoss
#        sce = softmaxcrossentropy_2d(x, labels, reduction='none')
#
#        # Check results
#        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_none')
#
#    @staticmethod
#    def export_softmaxcrossentropy_none_weights():
#        # Define operator attributes.
#        reduction = 'none'
#
#        # Create operator.
#        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
#                                     inputs=['x', 'y', 'w'],
#                                     outputs=['z'],
#                                     reduction=reduction)
#
#        # Define operator inputs.
#        np.random.seed(0)
#        x = np.random.rand(3, 5).astype(np.float32)
#        labels = np.random.randint(0, high=5, size=(3, ))
#        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)
#
#        # Compute SoftmaxCrossEntropyLoss
#        sce = softmaxcrossentropy_2d(x, labels, weight=weights, reduction='none')
#
#        # Check results
#        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_none_weights')
#
#    @staticmethod
#    def export_softmaxcrossentropy_sum():
#        # Define operator attributes.
#        reduction = 'sum'
#
#        # Create operator.
#        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
#                                     inputs=['x', 'y'],
#                                     outputs=['z'],
#                                     reduction=reduction)
#
#        # Define operator inputs.
#        np.random.seed(0)
#        x = np.random.rand(3, 5).astype(np.float32)
#        labels = np.random.randint(0, high=5, size=(3, ))
#
#        # Compute SoftmaxCrossEntropyLoss
#        sce = softmaxcrossentropy_2d(x, labels, reduction='sum')
#
#        # Check results
#        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_sum')
#
    @staticmethod
    def export_softmaxcrossentropy_mean():
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                     inputs=['x', 'y'],
                                     outputs=['z'],
                                     reduction=reduction)

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy_2d(x, labels)
        import ptvsd

        #5678 is the default attach port in the VS Code debug configurations
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name='test_softmax_cross_entropy_mean')
#
#    @staticmethod
#    def export_softmaxcrossentropy_mean_3d():
#        # Define operator attributes.
#        reduction = 'mean'
#
#        # Create operator.
#        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
#                                     inputs=['x', 'y'],
#                                     outputs=['z'],
#                                     reduction=reduction)
#
#        # Define operator inputs.
#        np.random.seed(0)
#        x = np.random.rand(3, 5, 2).astype(np.float32)
#        y = np.random.randint(0, high=5, size=(3, 2))
#
#        # Compute SoftmaxCrossEntropyLoss
#        sce = softmaxcrossentropy_3d(x, y)
#        import ptvsd
#
#        # 5678 is the default attach port in the VS Code debug configurations
#        print("Waiting for debugger attach")
#        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
#        ptvsd.wait_for_attach()
#        breakpoint()
#        # Check results
#        expect(node, inputs=[x, y], outputs=[sce], name='test_softmax_cross_entropy_mean_3d')
#    
#    @staticmethod
#    def export_softmaxcrossentropy_mean_weights():
#        # Define operator attributes.
#        reduction = 'mean'
#
#        # Create operator.
#        node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
#                                     inputs=['x', 'y', 'w'],
#                                     outputs=['z'],
#                                     reduction=reduction)
#
#        # Define operator inputs.
#        np.random.seed(0)
#        x = np.random.rand(3, 5).astype(np.float32)
#        labels = np.random.randint(0, high=5, size=(3, ))
#        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)
#
#        # Compute SoftmaxCrossEntropyLoss
#        sce = softmaxcrossentropy_2d(x, labels, weight=weights)
#
#        # Check results
#        expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_softmax_cross_entropy_mean_weight')
