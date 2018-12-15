from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class NonMaxSuppression(Base):

    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=3,
            iou_threshold=0.5,
            score_threshold=0.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0, 5]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU')

    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU_and_scores():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=3,
            iou_threshold=0.5,
            score_threshold=0.4

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU_and_scores')

    @staticmethod
    def export_nonmaxsuppression_suppress_zero_scores():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=6,
            iou_threshold=0.5,
            score_threshold=-3.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.1, 0.0, 0.0, 0.3, 0.2, -5.0]).astype(np.float32)
        selected_indices = np.array([3, 0]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_zero_scores')

    @staticmethod
    def export_nonmaxsuppression_flipped_coordinates():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=3,
            iou_threshold=0.5,
            score_threshold=-0.0

        )
        boxes = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, 0.9, 1.0, -0.1],
            [0.0, 10.0, 1.0, 11.0],
            [1.0, 10.1, 0.0, 11.1],
            [1.0, 101.0, 0.0, 100.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0, 5]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_flipped_coordinates')

    @staticmethod
    def export_nonmaxsuppression_limit_output_size():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=2,
            iou_threshold=0.5,
            score_threshold=0.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_limit_output_size')

    @staticmethod
    def export_nonmaxsuppression_max_output_size():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=30,
            iou_threshold=0.5,
            score_threshold=0.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0, 5]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_max_output_size')

    @staticmethod
    def export_nonmaxsuppression_single_box():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=3,
            iou_threshold=0.5,
            score_threshold=0.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0]
        ]).astype(np.float32)
        scores = np.array([0.9]).astype(np.float32)
        selected_indices = np.array([0]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')

    @staticmethod
    def export_nonmaxsuppression_identical_boxes():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices'],
            max_output_size=3,
            iou_threshold=0.5,
            score_threshold=0.0

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],

            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).astype(np.float32)
        selected_indices = np.array([0]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_identical_boxes')
        
    @staticmethod
    def export_nonmaxsuppression_pad_to_five_output():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices', 'valid_outputs'],
            max_output_size=5,
            iou_threshold=0.5,
            score_threshold=0.0,
            pad_to_max_output_size=1

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0, 5, 0, 0]).astype(np.int32)
        valid_outputs = np.array([3]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_pad_to_five_output')
        
    @staticmethod
    def export_nonmaxsuppression_pad_to_six_output():  # type: () -> None
        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['boxes', 'scores'],
            outputs=['selected_indices', 'valid_outputs'],
            max_output_size=6,
            iou_threshold=0.5,
            score_threshold=0.4,
            pad_to_max_output_size=1

        )
        boxes = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0]
        ]).astype(np.float32)
        scores = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3]).astype(np.float32)
        selected_indices = np.array([3, 0, 0, 0, 0, 0]).astype(np.int32)
        valid_outputs = np.array([2]).astype(np.int32)

        expect(node, inputs=[boxes, scores], outputs=[selected_indices], name='test_nonmaxsuppression_pad_to_six_output')

