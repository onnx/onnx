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

    pad_width = []
    for _ in range(input_rank):
        pad_width += [[0, 0]]  # init to zero

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]

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
    def export_constant_pad_negative_pads() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "value"], outputs=["y"], mode="constant"
        )
        x = np.array([1, 2, 3], dtype=np.int32)
        pads = np.array([-4, 2], dtype=np.int64)
        value = np.int32(-1)
        y = np.array([-1], dtype=np.int32)

        expect(
            node,
            inputs=[x, pads, value],
            outputs=[y],
            name="test_constant_pad_negative_pads",
        )

    @staticmethod
    def export_constant_pad_to_empty() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads"], outputs=["y"], mode="constant"
        )
        x = np.array([1, 2, 3], dtype=np.float32)
        pads = np.array([-3, 0], dtype=np.int64)
        y = np.empty((0,), dtype=np.float32)

        expect(
            node,
            inputs=[x, pads],
            outputs=[y],
            name="test_constant_pad_to_empty",
        )

    @staticmethod
    def export_negative_pad_axes() -> None:
        node = onnx.helper.make_node(
            "Pad", inputs=["x", "pads", "", "axes"], outputs=["y"], mode="edge"
        )
        x = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        pads = np.array([-1, 2], dtype=np.int64)
        axes = np.array([-1], dtype=np.int32)
        y = np.array(
            [[1, 2, 3, 3, 3], [5, 6, 7, 7, 7]],
            dtype=np.int64,
        )

        expect(
            node,
            inputs=[x, pads, axes],
            outputs=[y],
            name="test_negative_pad_axes",
        )

    @staticmethod
    def export_negative_pad_reflect_and_wrap() -> None:
        x = np.array([0, 1, 2, 3], dtype=np.float32)
        pads = np.array([-1, 2], dtype=np.int64)
        expected = {
            "reflect": np.array([1, 2, 3, 2, 1], dtype=np.float32),
            "wrap": np.array([1, 2, 3, 1, 2], dtype=np.float32),
        }

        for mode, y in expected.items():
            node = onnx.helper.make_node(
                "Pad", inputs=["x", "pads"], outputs=["y"], mode=mode
            )
            expect(
                node,
                inputs=[x, pads],
                outputs=[y],
                name=f"test_negative_pad_{mode}",
            )
