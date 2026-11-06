# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

_OPSET_IMPORTS = [onnx.helper.make_opsetid("", 29)]


class SearchSorted(Base):
    @staticmethod
    def export_searchsorted() -> None:
        node = onnx.helper.make_node(
            "SearchSorted",
            inputs=["x1", "x2"],
            outputs=["out"],
        )
        x1 = np.array([1, 3, 5, 7], dtype=np.float32)
        x2 = np.array([0, 3, 8], dtype=np.float32)
        out = np.searchsorted(x1, x2, side="left").astype(np.int64)
        # out == [0, 1, 4]

        expect(
            node,
            inputs=[x1, x2],
            outputs=[out],
            name="test_searchsorted",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_searchsorted_right() -> None:
        node = onnx.helper.make_node(
            "SearchSorted",
            inputs=["x1", "x2"],
            outputs=["out"],
            side="right",
        )
        x1 = np.array([1, 3, 5, 7], dtype=np.float32)
        x2 = np.array([0, 3, 8], dtype=np.float32)
        out = np.searchsorted(x1, x2, side="right").astype(np.int64)
        # out == [0, 2, 4]

        expect(
            node,
            inputs=[x1, x2],
            outputs=[out],
            name="test_searchsorted_right",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_searchsorted_sorter() -> None:
        node = onnx.helper.make_node(
            "SearchSorted",
            inputs=["x1", "x2", "sorter"],
            outputs=["out"],
        )
        # x1 is unsorted; sorter contains the indices that sort it ascending.
        x1 = np.array([5, 1, 7, 3], dtype=np.float32)
        sorter = np.argsort(x1).astype(np.int64)
        x2 = np.array([0, 3, 8], dtype=np.float32)
        out = np.searchsorted(x1, x2, side="left", sorter=sorter).astype(np.int64)

        expect(
            node,
            inputs=[x1, x2, sorter],
            outputs=[out],
            name="test_searchsorted_sorter",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_searchsorted_2d_values() -> None:
        node = onnx.helper.make_node(
            "SearchSorted",
            inputs=["x1", "x2"],
            outputs=["out"],
        )
        x1 = np.array([1, 3, 5, 7], dtype=np.float32)
        x2 = np.array([[0, 3], [4, 8]], dtype=np.float32)
        out = np.searchsorted(x1, x2, side="left").astype(np.int64)

        expect(
            node,
            inputs=[x1, x2],
            outputs=[out],
            name="test_searchsorted_2d_values",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_searchsorted_int64() -> None:
        node = onnx.helper.make_node(
            "SearchSorted",
            inputs=["x1", "x2"],
            outputs=["out"],
        )
        x1 = np.array([1, 3, 5, 7], dtype=np.int64)
        x2 = np.array([0, 3, 8], dtype=np.int64)
        out = np.searchsorted(x1, x2, side="left").astype(np.int64)

        expect(
            node,
            inputs=[x1, x2],
            outputs=[out],
            name="test_searchsorted_int64",
            opset_imports=_OPSET_IMPORTS,
        )

    @staticmethod
    def export_searchsorted_nan() -> None:
        # NaNs are ordered after all non-NaN values and are equivalent to each other.
        x1 = np.array([1, 2, 3, np.nan, np.nan], dtype=np.float32)
        x2 = np.array([2.5, np.nan, np.inf], dtype=np.float32)
        for side in ("left", "right"):
            node = onnx.helper.make_node(
                "SearchSorted",
                inputs=["x1", "x2"],
                outputs=["out"],
                side=side,
            )
            out = np.searchsorted(x1, x2, side=side).astype(np.int64)
            # left: out == [2, 3, 3]; right: out == [2, 5, 3]

            expect(
                node,
                inputs=[x1, x2],
                outputs=[out],
                name=f"test_searchsorted_nan_{side}",
                opset_imports=_OPSET_IMPORTS,
            )

    @staticmethod
    def export_searchsorted_signed_zero() -> None:
        # -0 and +0 are equivalent.
        x1 = np.array([-1, -0.0, 0.0, 1], dtype=np.float32)
        x2 = np.array([-0.0, 0.0], dtype=np.float32)
        for side in ("left", "right"):
            node = onnx.helper.make_node(
                "SearchSorted",
                inputs=["x1", "x2"],
                outputs=["out"],
                side=side,
            )
            out = np.searchsorted(x1, x2, side=side).astype(np.int64)
            # left: out == [1, 1]; right: out == [3, 3]

            expect(
                node,
                inputs=[x1, x2],
                outputs=[out],
                name=f"test_searchsorted_signed_zero_{side}",
                opset_imports=_OPSET_IMPORTS,
            )
