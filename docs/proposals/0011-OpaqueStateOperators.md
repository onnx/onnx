<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Opaque State Operators

## Status

This document proposes three operators in the prototype custom domain
`org.onnx.experimental.state`, opset 1:

- `AttentionWithCache`
- `LinearAttentionWithState`
- `CausalConvWithState`

The namespace is intentionally outside `ai.onnx`. The proposal is an
experiment and does not claim acceptance into the ONNX standard. Existing
tensor operators, including `ai.onnx.Attention`, `ai.onnx.LinearAttention`,
and `ai.onnx.CausalConvWithState`, remain unchanged.

## Motivation

Tensor state exposes a physical layout in the graph. That is useful for
portable tensor math, but it prevents the same graph from selecting a dense
cache, paged cache, ring buffer, compressed accumulator, or device-native
representation according to deployment policy.

ONNX Opaque values can instead name the logical kind of state while leaving
its physical representation to the implementation. State remains explicit:
every operator consumes an optional prior snapshot and returns a new snapshot.
There is no hidden mutation or process-global state.

## Common semantics

Each family has a distinct nominal Opaque type:

| Operator | State type |
|---|---|
| `AttentionWithCache` | `opaque(org.onnx.experimental.state,KVCache)` |
| `LinearAttentionWithState` | `opaque(org.onnx.experimental.state,LinearAttentionState)` |
| `CausalConvWithState` | `opaque(org.onnx.experimental.state,CausalConvState)` |

The nominal type identifies logical compatibility, not a physical layout,
runtime owner, allocation, or slot name. A runtime may additionally require
that a state value is consumed by the same session, device, or bound state
slot that produced it. Those are runtime binding constraints, not part of
ONNX type identity.

An omitted state input creates the operator-specific empty state. Input state
is a logical snapshot and is not mutated. Output state is a new logical
snapshot, even if an implementation aliases storage or uses copy-on-write.
Physical state policy is selected outside the graph, so one graph can execute
with different compatible policies.

`allow_lossy_state=0` requires lossless logical storage according to the
operator's tensor element types. Setting it to 1 opts into
implementation-selected lossy storage without selecting a representation in
the graph.

The initial runtime prototype validates only lossless, same-session,
matching-state-binding CPU snapshots. Paged storage, alternative device
placements, lossy storage, and all optional operator forms remain part of the
proposal but are not runtime-validated by that subset.

## AttentionWithCache

```text
AttentionWithCache(
    Q, K, V, attn_mask?, cache?, q_positions?, kv_positions?,
    nonpad_kv_seqlen?
) -> (Y, present_cache, qk_matmul_output?)
```

Attributes follow `ai.onnx.Attention`: `is_causal`, `scale`, `q_num_heads`,
`kv_num_heads`, `softmax_precision`, `softcap`, `left_window_size`, and
`right_window_size`, plus `allow_lossy_state`.

The logical `KVCache` is an ordered per-batch sequence of key/value records
and their absolute positions. Current valid K/V records append to the prior
sequence. Dense and paged caches are physically different representations of
the same logical state.

`q_positions` and `kv_positions` have shapes `(B, Sq)` and `(B, Skv)`.
When omitted, positions are the contiguous append after the prior cache.
`nonpad_kv_seqlen` has shape `(B)` and selects the valid prefix of the current
K/V chunk. Invalid tail records are not appended. Causal and window masks use
absolute positions. `attn_mask` is additionally applied over logical key order.

## LinearAttentionWithState

```text
LinearAttentionWithState(
    query, key, value, state?, decay?, beta?, sequence_lens?
) -> (output, present_state)
```

Attributes and recurrence follow `ai.onnx.LinearAttention`: `update_rule`,
`scale`, `q_num_heads`, `kv_num_heads`, `v_num_heads`, and `chunk_size`, plus
`allow_lossy_state`.

The logical state is the recurrent accumulator required by the selected
update rule. Omitted state is an all-zero accumulator. It is not a KV
sequence; implementations may use matrix, tiled, compressed, or device-native
representations without adopting paged-KV semantics.

`sequence_lens` has shape `(B)` and selects a valid prefix. Invalid tail tokens
produce zero output and do not update the accumulator.

## CausalConvWithState

```text
CausalConvWithState(
    input, weight, bias?, state?, sequence_lens?
) -> (output, present_state)
```

Attributes follow `ai.onnx.CausalConvWithState`: `activation` and `dilation`,
plus `allow_lossy_state`.

The logical state is the last `(K - 1) * dilation` raw input samples per batch
and channel in chronological order. Omitted state initializes this history to
zero. The state is bounded history, not a KV sequence; a ring buffer is one
natural physical representation.

`sequence_lens` has shape `(B)` and selects a valid prefix. Invalid tail
positions produce zero output and do not advance history.

## Alternatives

- **Expose tensor state only.** This is already supported by the existing
  operators but fixes physical layout in the graph.
- **Use one generic Opaque state type.** This weakens compatibility checking
  and permits accidental cross-family connections.
- **Apply paged-KV semantics to every operator.** Linear attention carries
  recurrent accumulators and causal convolution carries bounded history;
  forcing them into a KV abstraction obscures their logical math.
- **Hide state inside the runtime.** This introduces side effects and prevents
  explicit graph dataflow, snapshotting, branching, and state transfer.
