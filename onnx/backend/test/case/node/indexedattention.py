# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class IndexedAttention(Base):
    @staticmethod
    def export() -> None:
        # A learned indexer/TopK can supply this list independently of Q/K/V.
        node = helper.make_node("IndexedAttention", ["Q", "K", "V", "indices"], ["Y"])
        q = np.zeros((1, 1, 1, 1), dtype=np.float32)
        k = np.zeros((1, 1, 3, 1), dtype=np.float32)
        v = np.array([2, 4, 8], dtype=np.float32).reshape(1, 1, 3, 1)
        indices = np.array([[[[0, 2, -1]]]], dtype=np.int64)
        y = np.array([[[[5]]]], dtype=np.float32)
        expect(node, [q, k, v, indices], [y], name="test_indexedattention")

    @staticmethod
    def export_gqa_causal() -> None:
        node = helper.make_node(
            "IndexedAttention", ["Q", "K", "V", "indices", "candidate_mask"], ["Y"]
        )
        q = np.zeros((1, 4, 2, 1), dtype=np.float32)
        k = np.zeros((1, 2, 4, 1), dtype=np.float32)
        v = np.array([2, 4, 8, 16, 20, 40, 80, 160], dtype=np.float32).reshape(
            1, 2, 4, 1
        )
        indices = np.array([[[[3, 0, 2], [3, 1, -1]]]], dtype=np.int64)
        # Causality compares original positions, not the ordinal in the selected list.
        query_positions = np.array([1, 3], dtype=np.int64).reshape(1, 1, 2, 1)
        candidate_mask = indices <= query_positions
        y = np.array([2, 10, 2, 10, 20, 100, 20, 100], dtype=np.float32).reshape(
            1, 4, 2, 1
        )
        expect(
            node,
            [q, k, v, indices, candidate_mask],
            [y],
            name="test_indexedattention_gqa_causal",
        )

    @staticmethod
    def export_duplicates_and_empty_rows() -> None:
        node = helper.make_node("IndexedAttention", ["Q", "K", "V", "indices"], ["Y"])
        q = np.zeros((1, 1, 2, 1), dtype=np.float32)
        k = np.zeros((1, 1, 2, 1), dtype=np.float32)
        v = np.array([3, 12], dtype=np.float32).reshape(1, 1, 2, 1)
        indices = np.array([[[[0, 0, 1], [-1, 2, -2]]]], dtype=np.int64)
        y = np.array([6, 0], dtype=np.float32).reshape(1, 1, 2, 1)
        expect(
            node, [q, k, v, indices], [y], name="test_indexedattention_duplicates_empty"
        )

    @staticmethod
    def export_zero_candidates() -> None:
        node = helper.make_node("IndexedAttention", ["Q", "K", "V", "indices"], ["Y"])
        q = np.ones((1, 2, 1, 2), dtype=np.float32)
        k = np.empty((1, 1, 0, 2), dtype=np.float32)
        v = np.empty((1, 1, 0, 3), dtype=np.float32)
        indices = np.empty((1, 1, 1, 0), dtype=np.int64)
        y = np.zeros((1, 2, 1, 3), dtype=np.float32)
        expect(
            node, [q, k, v, indices], [y], name="test_indexedattention_zero_candidates"
        )

    @staticmethod
    def export_microblock_tail() -> None:
        node = helper.make_node("IndexedAttention", ["Q", "K", "V", "indices"], ["Y"])
        q = np.zeros((1, 1, 1, 1), dtype=np.float32)
        k = np.zeros((1, 1, 9, 1), dtype=np.float32)
        v = np.arange(9, dtype=np.float32).reshape(1, 1, 9, 1)
        # QSA groups compacted visible positions; the partial tail remains selected.
        visible = np.array([0, 2, 3, 5, 6, 8], dtype=np.int32)
        complete_groups = visible[:4].reshape(1, 4)
        selected_groups = complete_groups[np.array([0])]
        indices = np.concatenate((selected_groups.reshape(-1), visible[4:])).reshape(
            1, 1, 1, 6
        )
        y = np.array([[[[4.0]]]], dtype=np.float32)
        expect(
            node, [q, k, v, indices], [y], name="test_indexedattention_microblock_tail"
        )
