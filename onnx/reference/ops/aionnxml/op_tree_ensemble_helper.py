# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


class TreeEnsembleAttributes:
    def __init__(self):
        self._names = []

    def add(self, name, value):
        if not name.endswith("_as_tensor"):
            self._names.append(name)
        if isinstance(value, list):
            if name in {
                "base_values",
                "class_weights",
                "nodes_values",
                "nodes_hitrates",
            }:
                value = np.array(value, dtype=np.float32)
            elif name.endswith("as_tensor"):
                value = np.array(value)
        setattr(self, name, value)

    def __str__(self):
        rows = ["Attributes"]
        for name in self._names:
            if name.endswith("_as_tensor"):
                name = name.replace("_as_tensor", "")  # noqa: PLW2901
            rows.append(f"  {name}={getattr(self, name)}")
        return "\n".join(rows)


class TreeEnsemble:
    def __init__(self, **kwargs):
        self.atts = TreeEnsembleAttributes()

        for name, value in kwargs.items():
            self.atts.add(name, value)

        self.tree_ids = sorted(set(self.atts.nodes_treeids))
        self.root_index = {tid: len(self.atts.nodes_treeids) for tid in self.tree_ids}
        for index, tree_id in enumerate(self.atts.nodes_treeids):
            self.root_index[tree_id] = min(self.root_index[tree_id], index)
        self.node_index = {
            (tid, nid): i
            for i, (tid, nid) in enumerate(
                zip(self.atts.nodes_treeids, self.atts.nodes_nodeids)
            )
        }

    def __str__(self) -> str:
        rows = ["TreeEnsemble", f"root_index={self.root_index}", str(self.atts)]
        return "\n".join(rows)

    def leaf_index_tree(self, X: np.ndarray, tree_id: int) -> int:
        """Computes the leaf index for one tree."""
        index = self.root_index[tree_id]
        while self.atts.nodes_modes[index] != "LEAF":
            x = X[self.atts.nodes_featureids[index]]
            if np.isnan(x):
                r = self.atts.nodes_missing_value_tracks_true[index] >= 1
            else:
                rule = self.atts.nodes_modes[index]
                th = self.atts.nodes_values[index]
                if rule == "BRANCH_LEQ":
                    r = x <= th
                elif rule == "BRANCH_LT":
                    r = x < th
                elif rule == "BRANCH_GTE":
                    r = x >= th
                elif rule == "BRANCH_GT":
                    r = x > th
                elif rule == "BRANCH_EQ":
                    r = x == th
                elif rule == "BRANCH_NEQ":
                    r = x != th
                else:
                    raise ValueError(
                        f"Unexpected rule {rule!r} for node index {index}."
                    )
            nid = (
                self.atts.nodes_truenodeids[index]
                if r
                else self.atts.nodes_falsenodeids[index]
            )
            index = self.node_index[tree_id, nid]
        return index

    def leave_index_tree(self, X: np.ndarray) -> np.ndarray:
        """Computes the leaf index for all trees."""
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        outputs = []
        for row in X:
            outs = []
            for tree_id in self.tree_ids:
                outs.append(self.leaf_index_tree(row, tree_id))  # noqa: PERF401
            outputs.append(outs)
        return np.array(outputs)
