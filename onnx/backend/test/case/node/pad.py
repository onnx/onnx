# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)

    if num_axes * 2 != raw_pads.size:
        raise ValueError("The number of elements in raw_pads should be 2 * num_axes")

    # Separate negative (crop) and positive (pad) values
    # Negative pads crop first, then positive pads apply padding
    slices = [slice(None)] * input_rank  # Start with full slices for all axes
    pad_width = [[0, 0] for _ in range(input_rank)]  # Initialize with no padding
    
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        
        begin_pad = raw_pads[i]
        end_pad = raw_pads[i + num_axes]
        
        # Handle negative pads (cropping)
        start_crop = max(0, -begin_pad)
        end_crop = max(0, -end_pad)
        
        # Update slice for this axis
        # Note: If total cropping exceeds dimension size, the slice will be empty
        # which is allowed for constant mode
        dim_size = data.shape[axis]
        slice_start = start_crop
        slice_end = max(slice_start, dim_size - end_crop)  # Ensure slice_end >= slice_start
        slices[axis] = slice(slice_start, slice_end)
        
        # Only positive padding remains
        pad_width[axis] = [max(0, begin_pad), max(0, end_pad)]
    
    # Apply cropping first
    data = data[tuple(slices)]
    
    # Then apply padding
    if mode == "constant":
        return np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )

    return np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )


class Pad(Base):
    @staticmethod
    def export_constant_pad() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        y = pad_impl(x, pads, "constant", 1.2)

        expect(node, inputs=[x, pads, value], outputs=[y], name="test_constant_pad")

    @staticmethod
    def export_reflection_edge_and_wrap_pad() -> None:
        for mode in ("edge", "reflect", "wrap"):
            node = onnx.helper.make_node(
                "Pad", inputs=["x", "pads"], outputs=["y"], mode=mode
            )
            x = np.random.randn(1, 3, 4, 5).astype(np.int32)
            pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(
                np.int64
            )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            y = pad_impl(x, pads, mode)

            expect(node, inputs=[x, pads], outputs=[y], name=f"test_{mode}_pad")

    @staticmethod
    def export_constant_pad_axes() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value", "axes"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 3, 0, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        axes = np.array([1, 3], dtype=np.int64)
        y = pad_impl(
            x,
            pads,
            "constant",
            1.2,
            [1, 3],
        )

        expect(
            node,
            inputs=[x, pads, value, axes],
            outputs=[y],
            name="test_constant_pad_axes",
        )

    @staticmethod
    def export_constant_pad_negative_axes() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value", "axes"], outputs=["y"], mode="constant"
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.float32)
        pads = np.array([0, 3, 0, 4]).astype(
            np.int64
        )  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        value = np.float32(1.2)
        axes = np.array([-3, -1], dtype=np.int64)
        y = pad_impl(
            x,
            pads,
            "constant",
            1.2,
            [-3, -1],
        )

        expect(
            node,
            inputs=[x, pads, value, axes],
            outputs=[y],
            name="test_constant_pad_negative_axes",
        )

    @staticmethod
    def export_negative_pads_constant() -> None:
        """Test negative pads (cropping) with constant mode."""
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value"], outputs=["y"], mode="constant"
        )
        x = np.array(
            [
                [1.0, 1.2, 1.4, 1.6],
                [2.0, 2.2, 2.4, 2.6],
                [3.0, 3.2, 3.4, 3.6],
            ],
            dtype=np.float32,
        )
        pads = np.array([0, -1, 0, -1]).astype(
            np.int64
        )  # Remove first column (begin) and last column (end)
        value = np.float32(0.0)
        y = pad_impl(x, pads, "constant", 0.0)

        expect(
            node,
            inputs=[x, pads, value],
            outputs=[y],
            name="test_pad_negative_pads_constant",
        )

    @staticmethod
    def export_negative_pads_reflect() -> None:
        """Test negative pads (cropping) with reflect mode."""
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads"], outputs=["y"], mode="reflect"
        )
        x = np.array(
            [
                [1.0, 1.2, 1.4, 1.6],
                [2.0, 2.2, 2.4, 2.6],
                [3.0, 3.2, 3.4, 3.6],
            ],
            dtype=np.float32,
        )
        pads = np.array([0, -1, 0, 1]).astype(
            np.int64
        )  # Remove first column, add one at end
        y = pad_impl(x, pads, "reflect")

        expect(
            node, inputs=[x, pads], outputs=[y], name="test_pad_negative_pads_reflect"
        )

    @staticmethod
    def export_negative_pads_edge() -> None:
        """Test negative pads (cropping) with edge mode."""
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads"], outputs=["y"], mode="edge"
        )
        x = np.array(
            [
                [1.0, 1.2, 1.4],
                [2.0, 2.2, 2.4],
            ],
            dtype=np.float32,
        )
        pads = np.array([-1, 0, 0, 1]).astype(
            np.int64
        )  # Remove first row (begin), add one column at end
        y = pad_impl(x, pads, "edge")

        expect(
            node, inputs=[x, pads], outputs=[y], name="test_pad_negative_pads_edge"
        )
