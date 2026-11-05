# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from numpy.random import RandomState

from onnx.reference.op_run import OpRun


def _dropout(
    X: np.ndarray,
    drop_probability: float = 0.5,
    seed: int | None = None,
    training_mode: bool = False,
    return_mask: bool = False,
) -> tuple[np.ndarray]:
    if drop_probability == 0 or not training_mode:
        if return_mask:
            return X, np.ones(X.shape, dtype=bool)
        return (X,)

    rnd = RandomState(seed)
    mask = rnd.uniform(0, 1.0, X.shape) >= drop_probability
    scale = 1.0 / (1.0 - drop_probability)
    return (mask * X * scale, mask.astype(bool)) if return_mask else (mask * X * scale,)


class DropoutBase(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def _private_run(
        self,
        X: np.ndarray,
        seed: int | None = None,
        ratio: float = 0.5,
        training_mode: bool = False,
    ) -> tuple[np.ndarray]:
        return _dropout(
            X,
            ratio,
            seed=seed,
            return_mask=self.n_outputs == 2,
            training_mode=training_mode,
        )


class Dropout_7(DropoutBase):
    def _run(self, X, ratio=None):
        return self._private_run(X, ratio)


class Dropout_12(DropoutBase):
    def _run(self, *inputs, seed=None, training_mode=None):
        X = inputs[0]
        # Omitted optional inputs arrive as None, not truncated from inputs.
        ratio = 0.5 if len(inputs) <= 1 or inputs[1] is None else inputs[1]
        # In opset 12 training_mode is an attribute (passed as a keyword);
        # in opset >= 13 it is the third input.
        if len(inputs) > 2:
            training_mode = inputs[2]
        training_mode = training_mode is not None and bool(training_mode)
        return self._private_run(
            X,
            seed=seed,
            ratio=ratio,
            training_mode=training_mode,
        )
