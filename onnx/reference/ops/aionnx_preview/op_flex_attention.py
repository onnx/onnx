# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from onnx import TensorProto
from onnx.reference.op_run import OpRun

if TYPE_CHECKING:
    from collections.abc import Sequence

# Mapping from TensorProto data type to numpy dtype for softmax_precision.
_SOFTMAX_PRECISION_TO_NP_DTYPE: dict[int, np.dtype] = {
    int(TensorProto.FLOAT): np.dtype("float32"),
    int(TensorProto.FLOAT16): np.dtype("float16"),
    int(TensorProto.DOUBLE): np.dtype("float64"),
}


def _call_mod_graph(
    evaluator: Any,
    positional_inputs: Sequence[np.ndarray],
    *,
    attributes: dict[str, Any] | None = None,
) -> np.ndarray:
    input_names = list(evaluator.input_names)
    if len(input_names) != len(positional_inputs):
        raise RuntimeError(
            f"Graph attribute expects {len(input_names)} inputs "
            f"but got {len(positional_inputs)}."
        )
    feeds = dict(zip(input_names, positional_inputs, strict=False))
    outs = evaluator.run(None, feeds, attributes=attributes)
    if not isinstance(outs, list) or len(outs) != 1:
        raise RuntimeError("Graph attribute must produce exactly 1 output.")
    return outs[0]


def _get_softmax_dtype(
    input_dtype: np.dtype,
    softmax_precision: int | None = None,
) -> np.dtype:
    """Determine the numpy dtype for softmax computation.

    Priority:
      1. Use explicit softmax_precision if provided.
      2. Promote float16 to float32 for numerical stability.
      3. Use input dtype as-is for float32/float64.
    """
    if softmax_precision is not None:
        if softmax_precision not in _SOFTMAX_PRECISION_TO_NP_DTYPE:
            raise ValueError(
                f"Unsupported softmax_precision value: {softmax_precision}"
            )
        return _SOFTMAX_PRECISION_TO_NP_DTYPE[softmax_precision]
    if input_dtype == np.float16:
        return np.dtype("float32")
    return np.dtype("float32") if input_dtype.itemsize < 4 else input_dtype


def _compute_flex_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float | None = None,
    score_mod: Any = None,
    prob_mod: Any = None,
    softmax_precision: int | None = None,
) -> tuple[np.ndarray,]:
    assert len(Q.shape) == len(K.shape) == len(V.shape) == 4

    _B, Hq, _L, E = Q.shape

    # Calculate scaling factor if not provided (default: 1/sqrt(head_size))
    if scale is None:
        scale = 1.0 / np.sqrt(E)
    _Bk, Hkv, S, Ek = K.shape
    _Bv, _Hkv2, Sv, _Ev = V.shape
    if Hkv != _Hkv2:
        raise RuntimeError("Key and value must share the same head dimension.")
    if Sv != S:
        raise RuntimeError("Key and value must share the same sequence length.")
    if Ek != E:
        raise RuntimeError("Query and key must share the same embedding dimension.")
    # Determine the computation dtype based on softmax_precision.
    compute_dtype = _get_softmax_dtype(Q.dtype, softmax_precision)
    Q_f = Q.astype(compute_dtype, copy=False)
    K_f = K.astype(compute_dtype, copy=False)
    V_f = V.astype(compute_dtype, copy=False)

    if Hq != Hkv:
        if Hkv <= 0 or (Hq % Hkv) != 0:
            raise RuntimeError(
                "q_num_heads must be a multiple of kv_num_heads when they differ."
            )
        repeat = Hq // Hkv
        K_f = np.repeat(K_f, repeats=repeat, axis=1)
        V_f = np.repeat(V_f, repeats=repeat, axis=1)

    # Scores: (B, Hq, L, S)
    scores = np.matmul(Q_f, np.swapaxes(K_f, -1, -2)) * float(scale)

    if score_mod is not None:
        scores_out = _call_mod_graph(score_mod, [scores])
        scores = np.asarray(scores_out, dtype=compute_dtype)

    # Softmax over kv dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    if prob_mod is not None:
        probs_out = _call_mod_graph(prob_mod, [probs])
        probs = np.asarray(probs_out, dtype=compute_dtype)

    out = np.matmul(probs, V_f)

    return (out.astype(V.dtype),)


class FlexAttention(OpRun):
    op_domain = "ai.onnx.preview"

    def _run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: float | None = None,
        score_mod: Any = None,
        prob_mod: Any = None,
        softmax_precision: int | None = None,
        **_: Any,
    ) -> tuple[np.ndarray]:
        return _compute_flex_attention(
            Q,
            K,
            V,
            scale=scale,
            score_mod=score_mod,
            prob_mod=prob_mod,
            softmax_precision=softmax_precision,
        )
