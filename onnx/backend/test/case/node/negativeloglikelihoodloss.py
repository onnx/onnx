from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def compute_negative_log_likelihood_loss(input, target, weight=None, reduction='mean'):  # type: ignore
    ''' Compute negative_log_likelihood_loss '''
    input_shape = input.shape

    # GatherElement(-input, target)
    if len(input_shape) == 2:
        N, C = input_shape
        neg_gather_element_input = np.zeros((N, ), dtype=np.float32)
        for i in range(N):
            neg_gather_element_input[i] = -input[i][target[i]]
    else:
        N, C, dim1, dim2 = input_shape
        neg_gather_element_input = np.zeros((N, dim1, dim2), dtype=np.float32)
        for i in range(N):
            for d1 in range(dim1):
                for d2 in range(dim2):
                    neg_gather_element_input[i][d1][d2] = -input[i][target[i][d1][d2]][d1][d2]

    loss = neg_gather_element_input
    if weight is not None:
        # Gather(input=weight, index=target)
        gather_weight = np.take(weight, target)

        loss = gather_weight * loss
        if reduction == 'mean':
            return loss.sum() / gather_weight.sum()

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return np.mean(loss, keepdims=False)
    elif reduction == 'sum':
        return np.sum(loss, keepdims=False)


class NegativeLogLikelihoodLoss(Base):

    @staticmethod
    def export_input_shape_is_NC():  # type: () -> None
        reduction = 'none'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C = 3, 5
        np.random.seed(0)
        input = np.random.rand(N, C).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, ))

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

        expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NC')

    @staticmethod
    def export_input_shape_is_NCd1d2():  # type: () -> None
        reduction = 'none'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

        expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2')

    @staticmethod
    def export_input_shape_is_NCd1d2_reduction_mean():  # type: () -> None
        reduction = 'mean'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

        expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean')

    @staticmethod
    def export_input_shape_is_NCd1d2_reduction_sum():  # type: () -> None
        reduction = 'sum'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

        expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum')

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight():  # type: () -> None
        reduction = 'none'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

        expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight')

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight_reduction_mean():  # type: () -> None
        reduction = 'mean'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

        expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean')

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight_reduction_sum():  # type: () -> None
        reduction = 'sum'
        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['input', 'target', 'weight'],
            outputs=['loss'],
            reduction=reduction
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2))
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

        expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
            name='test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum')
