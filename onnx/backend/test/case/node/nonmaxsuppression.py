# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class NonMaxSuppression(Base):
    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_suppress_by_IOU",
        )

    @staticmethod
    def export_nonmaxsuppression_suppress_by_IOU_and_scores() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.4]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_suppress_by_IOU_and_scores",
        )

    @staticmethod
    def export_nonmaxsuppression_flipped_coordinates() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, 0.9, 1.0, -0.1],
                    [0.0, 10.0, 1.0, 11.0],
                    [1.0, 10.1, 0.0, 11.1],
                    [1.0, 101.0, 0.0, 100.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_flipped_coordinates",
        )

    @staticmethod
    def export_nonmaxsuppression_limit_output_size() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_limit_output_size",
        )

    @staticmethod
    def export_nonmaxsuppression_single_box() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
        scores = np.array([[[0.9]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_single_box",
        )

    @staticmethod
    def export_nonmaxsuppression_identical_boxes() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array(
            [[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]
        ).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_identical_boxes",
        )

    @staticmethod
    def export_nonmaxsuppression_center_point_box_format() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
            center_point_box=1,
        )
        boxes = np.array(
            [
                [
                    [0.5, 0.5, 1.0, 1.0],
                    [0.5, 0.6, 1.0, 1.0],
                    [0.5, 0.4, 1.0, 1.0],
                    [0.5, 10.5, 1.0, 1.0],
                    [0.5, 10.6, 1.0, 1.0],
                    [0.5, 100.5, 1.0, 1.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_center_point_box_format",
        )

    @staticmethod
    def export_nonmaxsuppression_two_classes() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ]
            ]
        ).astype(np.float32)
        scores = np.array(
            [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]
        ).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array(
            [[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]
        ).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_two_classes",
        )

    @staticmethod
    def export_nonmaxsuppression_two_batches() -> None:
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ],
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [0.0, -0.1, 1.0, 0.9],
                    [0.0, 10.0, 1.0, 11.0],
                    [0.0, 10.1, 1.0, 11.1],
                    [0.0, 100.0, 1.0, 101.0],
                ],
            ]
        ).astype(np.float32)
        scores = np.array(
            [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]], [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]
        ).astype(np.float32)
        max_output_boxes_per_class = np.array([2]).astype(np.int64)
        iou_threshold = np.array([0.5]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        selected_indices = np.array(
            [[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]
        ).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_two_batches",
        )

    @staticmethod
    def export_nonmaxsuppression_iou_threshold_boundary() -> None:
        """Test boundary condition where IoU exactly equals threshold.
        
        This test verifies that the comparison is strict (>), not inclusive (>=).
        When IoU exactly equals the threshold, boxes should be KEPT, not suppressed.
        This follows PyTorch's NMS implementation.
        """
        node = onnx.helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
        )
        # Two boxes with 50% overlap: box1=[0,0,1,1], box2=[0.5,0.5,1.5,1.5]
        # Intersection area = 0.5 * 0.5 = 0.25
        # Union area = 1.0 + 1.0 - 0.25 = 1.75
        # IoU = 0.25 / 1.75 = 0.142857... ≈ 0.1429
        # Setting threshold to exactly this value tests the boundary
        boxes = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],  # box 0
                    [0.5, 0.5, 1.5, 1.5],  # box 1 - overlaps box 0 with IoU ~0.1429
                ]
            ]
        ).astype(np.float32)
        scores = np.array([[[0.9, 0.8]]]).astype(np.float32)
        max_output_boxes_per_class = np.array([3]).astype(np.int64)
        # Set threshold to approximately the IoU of the two boxes
        # With strict >, both boxes should be selected since IoU is not > threshold
        iou_threshold = np.array([0.142857]).astype(np.float32)
        score_threshold = np.array([0.0]).astype(np.float32)
        # Both boxes should be selected because IoU ~= threshold, not > threshold
        selected_indices = np.array([[0, 0, 0], [0, 0, 1]]).astype(np.int64)

        expect(
            node,
            inputs=[
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
            ],
            outputs=[selected_indices],
            name="test_nonmaxsuppression_iou_threshold_boundary",
        )
