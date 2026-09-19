# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class Normalizer(OpRunAiOnnxMl):
    @staticmethod
    def _scaled(x):
        x_float = x.astype(np.float64, copy=False)
        axis = 0 if x.ndim == 1 else 1
        scale = np.max(np.abs(x_float), axis=axis, keepdims=True)
        return np.divide(
            x_float,
            scale,
            out=x_float.copy(),
            where=scale != 0,
        )

    @staticmethod
    def norm_max(x):
        """Max normalization"""
        return Normalizer._scaled(x).astype(np.float32)

    @staticmethod
    def norm_l1(x):
        """L1 normalization"""
        scaled = Normalizer._scaled(x)
        axis = 0 if x.ndim == 1 else 1
        norm = np.sum(np.abs(scaled), axis=axis, keepdims=True)
        return np.divide(
            scaled,
            norm,
            out=scaled.copy(),
            where=norm != 0,
        ).astype(np.float32)

    @staticmethod
    def norm_l2(x):
        """L2 normalization"""
        scaled = Normalizer._scaled(x)
        axis = 0 if x.ndim == 1 else 1
        norm = np.sqrt(np.sum(np.square(scaled), axis=axis, keepdims=True))
        return np.divide(
            scaled,
            norm,
            out=scaled.copy(),
            where=norm != 0,
        ).astype(np.float32)

    def _run(self, x, norm=None):
        if norm == "MAX":
            _norm = Normalizer.norm_max
        elif norm == "L1":
            _norm = Normalizer.norm_l1
        elif norm == "L2":
            _norm = Normalizer.norm_l2
        else:
            raise ValueError(f"Unexpected value for norm='{norm}'.")
        return (_norm(x),)
