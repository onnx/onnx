# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize("batch_index", [-1, 2])
def test_max_roi_pool_rejects_out_of_range_batch_index(batch_index: int) -> None:
    node = make_node(
        "MaxRoiPool",
        ["X", "rois"],
        ["Y"],
        pooled_shape=[1, 1],
        spatial_scale=1.0,
    )
    sess = ReferenceEvaluator(node)

    with pytest.raises(
        ValueError, match=f"ROI batch index {batch_index} is out of range"
    ):
        sess.run(
            None,
            {
                "X": np.zeros((2, 1, 1, 1), dtype=np.float32),
                "rois": np.array([[batch_index, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            },
        )
