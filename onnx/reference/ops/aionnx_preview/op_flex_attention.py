# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_attention import _softmax

if TYPE_CHECKING:
    from collections.abc import Sequence


def _to_scalar_tensor_i64(v: int) -> np.ndarray:
    # Indices are scalar tensors (0-d arrays)
    return np.array(v, dtype=np.int64)


def _to_scalar_tensor_f32(v: float) -> np.ndarray:
    return np.array(v, dtype=np.float32)


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


def _compute_flex_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float | None = None,
    enable_gqa: int | None = None,
    mask_value: float = -np.inf,
    score_mod: Any = None,
    mask_mod: Any = None,
    prob_mod: Any = None,
) -> tuple[np.ndarray,]:
    assert len(Q.shape) == len(K.shape) == len(V.shape) == 4

    B, Hq, L, E = Q.shape

    # Calculate scaling factor if not provided (default: 1/sqrt(head_size))
    if scale is None:
        scale = 1.0 / np.sqrt(E)
    _Bk, Hkv, S, _Ek = K.shape
    _Bv, _Hkv2, _Sv, Ev = V.shape
    # Compute in float32 for stability/portability.
    Q_f = Q.astype(np.float32, copy=False)
    K_f = K.astype(np.float32, copy=False)
    V_f = V.astype(np.float32, copy=False)

    group = (Hq // Hkv) if enable_gqa == 1 else 1
    out = np.empty((B, Hq, L, Ev), dtype=np.float32)

    # Main loops: b, hq, q_idx
    for b in range(B):
        for hq in range(Hq):
            kvh = (hq // group) if enable_gqa == 1 else hq

            # Slice K,V once per head for locality
            K_mat = K_f[b, kvh, :, :]  # (S, E)
            V_mat = V_f[b, kvh, :, :]  # (S, Ev)

            for q_idx in range(L):
                q_vec = Q_f[b, hq, q_idx, :]  # (E,)

                # raw scores: (S,)
                scores = (K_mat @ q_vec) * float(scale)  # float32

                # Apply score_mod / mask_mod elementwise if provided
                if score_mod is not None or mask_mod is not None:
                    scores2 = np.empty((S,), dtype=np.float32)
                    for kv_idx in range(S):
                        s = float(scores[kv_idx])

                        if score_mod is not None:
                            # (score, batch, head, q_idx, kv_idx) -> score_out
                            pos = [
                                _to_scalar_tensor_f32(s),
                                _to_scalar_tensor_i64(b),
                                _to_scalar_tensor_i64(hq),
                                _to_scalar_tensor_i64(q_idx),
                                _to_scalar_tensor_i64(kv_idx),
                            ]
                            s_out = _call_mod_graph(score_mod, pos)
                            s = float(np.asarray(s_out, dtype=np.float32).reshape(()))

                        if mask_mod is not None:
                            # (batch, head, q_idx, kv_idx) -> bool
                            pos = [
                                _to_scalar_tensor_i64(b),
                                _to_scalar_tensor_i64(hq),
                                _to_scalar_tensor_i64(q_idx),
                                _to_scalar_tensor_i64(kv_idx),
                            ]
                            m_out = _call_mod_graph(mask_mod, pos)
                            m = bool(np.asarray(m_out).reshape(()))
                            if not m:
                                s = float(mask_value)

                        scores2[kv_idx] = np.float32(s)
                    scores = scores2

                # Softmax over kv dimension
                probs = _softmax(scores)

                # Apply prob_mod elementwise if provided
                if prob_mod is not None:
                    probs2 = np.empty((S,), dtype=np.float32)
                    for kv_idx in range(S):
                        p = float(probs[kv_idx])
                        # (prob, batch, head, q_idx, kv_idx) -> prob_out
                        pos = [
                            _to_scalar_tensor_f32(p),
                            _to_scalar_tensor_i64(b),
                            _to_scalar_tensor_i64(hq),
                            _to_scalar_tensor_i64(q_idx),
                            _to_scalar_tensor_i64(kv_idx),
                        ]
                        p_out = _call_mod_graph(prob_mod, pos)
                        p = float(np.asarray(p_out, dtype=np.float32).reshape(()))
                        probs2[kv_idx] = np.float32(p)
                    probs = probs2

                # Y = probs @ V
                out[b, hq, q_idx, :] = probs @ V_mat  # (Ev,)

    return (out.astype(V.dtype),)


class FlexAttention(OpRun):
    op_domain = "ai.onnx.preview"

    def _run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: float | None = None,
        enable_gqa: int | None = None,
        mask_value: float = -np.inf,
        score_mod: Any = None,
        mask_mod: Any = None,
        prob_mod: Any = None,
        **_: Any,
    ) -> tuple[np.ndarray]:
        return _compute_flex_attention(
            Q,
            K,
            V,
            scale=scale,
            enable_gqa=enable_gqa,
            mask_value=mask_value,
            score_mod=score_mod,
            mask_mod=mask_mod,
            prob_mod=prob_mod,
        )
