<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-linear-attention-prefix-state)=

# Accepted-Prefix State Recovery for LinearAttention

A model that verifies `T` candidate tokens in one `LinearAttention` call may
keep only the first `a` of them. The next call then needs the recurrent state
after exactly `a` tokens. This page describes a standard ONNX graph for that
selection and an optimization a runtime can apply to it without changing the
graph's results. It uses only opset-29 `LinearAttention` and existing
operators; no schema change is involved.

ONNX defines the values a graph computes. How a runtime represents
intermediate values is an implementation choice, as long as the results the
graph exposes are unchanged.

## Canonical graph

```
output, present_state, states =
    LinearAttention<q_num_heads, kv_num_heads, v_num_heads, update_rule, scale>(
        query, key, value, past_state, decay, beta)
initial    = Unsqueeze(past_state, axes=[1])          # [B, 1, H_v, d_k, d_v]
trajectory = Concat<axis=1>(initial, states)          # [B, T+1, H_v, d_k, d_v]
```

`trajectory[:, i]` is the state after `i` tokens: `i = 0` is the initial
state and `i = T` is `present_state`. Select the accepted state with either:

```
# Shared acceptance count: accepted is an int64 scalar in [0, T].
accepted_state = Gather<axis=1>(trajectory, accepted)            # [B, H_v, d_k, d_v]

# Per-batch acceptance counts: accepted is int64 [B], each in [0, T].
index          = Unsqueeze(accepted, axes=[1])                   # [B, 1]
accepted_state = GatherND<batch_dims=1>(trajectory, index)       # [B, H_v, d_k, d_v]
```

Do not use `Gather<axis=1>` with a `[B]` index: it selects every index for
every batch row and produces `[B, B, H_v, d_k, d_v]`.

| Tensor | Shape | Type |
|---|---|---|
| `query`, `key` | `[B, T, H_q*d_k]`, `[B, T, H_kv*d_k]` | `T` |
| `value` | `[B, T, H_v*d_v]` | `T` |
| `decay` (gated modes) | `[B, T, H_v]` or `[B, T, H_v*d_k]` | `T` |
| `beta` (delta modes) | `[B, T, H_v]` or `[B, T, 1]` | `T` |
| `output` | `[B, T, H_o*d_v]` | `T` |
| `past_state`, `present_state`, `accepted_state` | `[B, H_v, d_k, d_v]` | `S` |
| `states` | `[B, T, H_v, d_k, d_v]` | `S` |

`H_v` is `v_num_heads` when it is positive and `kv_num_heads` otherwise.
`S` may differ from `T`, for example a float32 state with float16
activations.

Notes:

- Supply `past_state` explicitly, using zeros for a fresh sequence. This
  makes `a = 0` well defined and gives `initial` the same type `S` as
  `states`, which `Concat` requires. When `past_state` is omitted, the state
  outputs take type `T`.
- `T` may be dynamic. For `T = 0`, `states` is `[B, 0, H_v, d_k, d_v]`,
  `trajectory` holds only the initial state, and only `a = 0` is valid.
- `Gather` and `GatherND` accept negative indices, which count from the end.
  Keep acceptance counts in `[0, T]`.
- `output` is causal, so each batch row's first `a` tokens of `output` are
  already the output of its accepted prefix and need no recovery.

The following example builds the per-batch form with float16 activations and
a float32 state and checks every selected state against a separate
`LinearAttention` call on the accepted prefix.

```{eval-rst}
.. exec_code::

    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.parser.parse_model("""
    <ir_version: 13, opset_import: ["" : 29]>
    recover (float16[B, T, 4] query, float16[B, T, 4] key, float16[B, T, 6] value,
             float[B, 2, 2, 3] past_state, float16[B, T, 2] decay,
             float16[B, T, 2] beta, int64[B] accepted)
        => (float16[B, T, 6] output, float[B, 2, 2, 3] accepted_state) {
        output, present_state, states = LinearAttention <
            q_num_heads: int = 2, kv_num_heads: int = 2, update_rule: string = "gated_delta"
        > (query, key, value, past_state, decay, beta)
        axis = Constant <value_ints: ints = [1]> ()
        initial = Unsqueeze (past_state, axis)
        trajectory = Concat <axis: int = 1> (initial, states)
        index = Unsqueeze (accepted, axis)
        accepted_state = GatherND <batch_dims: int = 1> (trajectory, index)
    }
    """)
    onnx.checker.check_model(model, full_check=True)

    rng = np.random.default_rng(0)
    B, T = 3, 4
    key = rng.normal(size=(B, T, 2, 2))
    key /= np.linalg.norm(key, axis=-1, keepdims=True)
    feeds = {
        "query": rng.normal(size=(B, T, 4)).astype(np.float16),
        "key": key.reshape(B, T, 4).astype(np.float16),
        "value": rng.normal(size=(B, T, 6)).astype(np.float16),
        "past_state": rng.normal(size=(B, 2, 2, 3)).astype(np.float32),
        "decay": -rng.uniform(0.1, 0.9, (B, T, 2)).astype(np.float16),
        "beta": rng.uniform(0.1, 0.9, (B, T, 2)).astype(np.float16),
        "accepted": np.array([0, 2, T], dtype=np.int64),
    }
    (selected,) = ReferenceEvaluator(model).run(["accepted_state"], feeds)

    prefix_model = onnx.parser.parse_model("""
    <ir_version: 13, opset_import: ["" : 29]>
    prefix (float16[1, A, 4] query, float16[1, A, 4] key, float16[1, A, 6] value,
            float[1, 2, 2, 3] past_state, float16[1, A, 2] decay, float16[1, A, 2] beta)
        => (float[1, 2, 2, 3] present_state) {
        output, present_state = LinearAttention <
            q_num_heads: int = 2, kv_num_heads: int = 2, update_rule: string = "gated_delta"
        > (query, key, value, past_state, decay, beta)
    }
    """)
    prefix = ReferenceEvaluator(prefix_model)
    for b, a in enumerate(feeds["accepted"]):
        inputs = {
            name: value[b : b + 1] if name == "past_state" else value[b : b + 1, :a]
            for name, value in feeds.items()
            if name != "accepted"
        }
        (expected,) = prefix.run(None, inputs)
        np.testing.assert_allclose(selected[b], expected[0], rtol=1e-6, atol=1e-6)
    print("accepted:", feeds["accepted"], "selected:", selected.shape, selected.dtype)
```

## Replaying accepted updates

Per state head, with `S` the preceding float32 `[d_k, d_v]` state, `k` the
mapped key vector, and `v` the value vector, opset-29 `LinearAttention`
applies:

| `update_rule` | Decayed state `D` | Write vector `u` | New state |
|---|---|---|---|
| `linear` | `S` | `v` | `D + k u^T` |
| `gated` | `exp(decay) * S` | `v` | `D + k u^T` |
| `delta` | `S` | `beta * (v - D^T k)` | `D + k u^T` |
| `gated_delta` | `exp(decay) * S` | `beta * (v - D^T k)` | `D + k u^T` |

Decay is log-space. A per-head decay is a scalar; a per-key decay has length
`d_k`, broadcasts over `d_v`, and scales rows of `S`. Each step is therefore
`S_t = gamma_t * S_{t-1} + k_t u_t^T`, with `gamma_t = exp(decay_t)` in gated
modes and `1` otherwise.
The operator normalizes neither queries nor keys, and `scale` affects only
`output`, not the state.

In the additive modes, `u_t` is the `value` input. In the delta modes, `u_t`
is a corrected write computed against the decayed state of the same prefix,
so it depends only on tokens `0..t` and is valid for every acceptance count
`a > t`. A runtime that keeps, per token and state head:

- the initial state, as `float32(past_state)`;
- `k_t`, from the `key` input and the head mapping;
- `gamma_t` or the `decay` input, in gated modes;
- `u_t`, which is the `value` input in additive modes and a float32 tensor
  `[B, T, H_v, d_v]` in delta modes,

can rebuild the accepted state by replaying only the first `a` updates:

```
# Forward pass: compute output and present_state as usual.
# In delta modes, also record u[b, t, h] in float32. Do not write `states`.
# gamma(b, t, h) is exp(decay) as a [d_k, 1] column (a per-head scalar is
# broadcast), or 1 in ungated modes.

def recover(a):                               # a: [B], each in [0, T]
    for b, h:
        X = float32(past_state[b, h])         # [d_k, d_v]
        for t in range(a[b]):
            k = key_vector(b, t, kv_head(h))  # float32 [d_k]
            X = gamma(b, t, h) * X + outer(k, u[b, t, h])   # gamma: [d_k, 1]
        accepted_state[b, h] = cast_to_S(X)
```

`a = 0` and `a = T` can return `past_state` and `present_state` directly.
The same pattern applies to every update rule and needs no knowledge of the
model architecture.

Cost, per call:

| | Storage | Work |
|---|---|---|
| Dense `states` | `B*T*H_v*d_k*d_v` elements of `S` | write every snapshot |
| Replay, additive modes | inputs already present | `a*H_v*d_k*d_v` state-element updates per batch row |
| Replay, delta modes | `B*T*H_v*d_v` float32 plus inputs | same |

In the delta modes, the retained data is smaller than the trajectory by
roughly a factor of `d_k`, adjusted for element sizes. Each state-element
update is one multiply-add, plus a decay multiply in gated modes, so replay
costs about one state update per accepted token. Dense snapshots can be
cheaper when `d_k` is small or when every snapshot is consumed.

## Where the optimization applies

The graph output contract determines what a runtime may change.

1. **Within one graph.** When `states` and `trajectory` feed only the
   selection and neither is a graph output, the runtime sees every consumer
   and can lower `LinearAttention` plus the selection to the forward pass and
   replay above.
2. **An engine that owns a cross-call pipeline.** Acceptance is often known
   only after the call returns. A runtime component that runs both the
   verification call and the next call, and never hands the trajectory to
   other code, can keep the retained data across calls in a private form and
   replay when the acceptance counts arrive.
3. **`states` as a graph output.** A graph output cannot reveal its future
   consumer, so a runtime asked for `states` must produce a dense
   `[B, T, H_v, d_k, d_v]` tensor of type `S`. A backend may offer a lazy or
   provider-owned value that materializes on access, but ONNX does not
   require one. Replacing the output with a packed tensor of a different
   shape or type changes the model's interface. Issue
   [#8831](https://github.com/onnx/onnx/issues/8831) discusses adding compact
   outputs to the operator; this guide relies on no such change.

## Matching conditions

A rewrite of the canonical graph needs to check that:

- `states` and `trajectory` have no consumers besides the selection and are
  not graph outputs. Other uses of the full trajectory need materialization.
- `initial` is `Unsqueeze(past_state, [1])` of the same tensor passed to
  `LinearAttention`, concatenated first along axis 1.
- The selection is `Gather<axis=1>` with a scalar index, or
  `GatherND<batch_dims=1>` with `[B, 1]` indices. Acceptance counts are in
  `[0, T]`, or the rewrite reproduces the operators' negative-index and
  out-of-range behavior.
- `past_state`, `key`, `decay`, and `beta`, and `value` in additive modes,
  stay alive and unmodified until replay, unless the runtime keeps its own
  float32 copy. Retaining `past_state` conflicts with reusing its buffer for
  `present_state`.
- `output` and `present_state` and their consumers are unchanged.
- Replay uses the same head mapping as the operator: with `v_num_heads = 0`,
  state head `h` uses key head `h`; with a positive `v_num_heads`, it uses
  key head `floor(h / (H_v / H_kv))`. Query grouping affects only `output`.
- Per-head and per-key decay and per-head and shared beta keep their
  broadcast shapes.
- A `Cast` after the selection is applied to the replayed value after the
  rounding to `S`.

If any condition fails, materialize `states`. Another standard alternative is
to run `LinearAttention` again on the accepted prefix of the inputs, which
recomputes the delta corrections instead of storing them.

## Numerics

- The recurrence runs in float32. Each snapshot in `states` is rounded to `S`,
  but that rounding does not feed back into later tokens. Replay must start
  from `float32(past_state)`, use float32 factors, and round to `S` once at
  the end. Never continue from a rounded snapshot.
- Retained factors must keep the precision the recurrence used. The `key`,
  `value`, `decay`, and `beta` inputs convert to float32 exactly, so keeping
  the original inputs is lossless. Delta-mode `u_t` is a float32 result and
  must keep float32 precision, even when `T` or `S` is narrower.
- Rebuild the accepted state forward from the initial state. Do not undo
  updates from `present_state`: dividing by `gamma_t` fails when it underflows
  to zero, and subtracting writes amplifies rounding error.
- Chunked or parallel kernels regroup sums, so they are not bitwise identical
  to a sequential recurrence. Replay should meet the same numerical
  expectations as the operator's reference semantics, not a bitwise guarantee
  across kernels.
