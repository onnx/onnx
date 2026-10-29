<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-attention-sinks)=

# Attention sinks with existing operators

Some models add a learned, denominator-only candidate to attention softmax.
For example, [GPT-OSS's attention implementation](https://github.com/openai/gpt-oss/blob/243a1b02767da73bd2e3975be250afa801635866/gpt_oss/torch/model.py#L167-L218)
has one learned sink logit per **query head**. It concatenates that logit after
scaling and masking the real-token logits, applies softmax, and discards the
sink probability before multiplying by real values.

This guide describes an explanatory composition of existing standard-domain
operators, not a change to the normative
{ref}`Attention operator <l-onnx-doc-Attention>`. It introduces no operator,
function registration, or runtime API. Here, a *sink* is a virtual softmax
candidate, not a real prefix token retained in a streaming KV cache.

## Computation and layout

Let `L` be the final real-token logits with shape `[B, Hq, Q, K]`, and `V` the
real values with shape `[B, Hq, K, Dv]`. `B` is batch size, `Hq` query-head
count, `Q` query length, and `K` the number of real key positions. For each
row, a finite sink logit `s` gives

$$
p_j = \frac{\exp(L_j)}{\exp(s) + \sum_{i=0}^{K-1}\exp(L_i)}, \qquad
Y = \sum_{j=0}^{K-1}p_j V_j.
$$

The sink is in **final-logit units**: do not multiply it by the query/key
scale, apply real-token softcap to it, or include it in a real-token mask.
Construct `L` first, preserving the model's operation order. A common order is
`MatMul(Q, Transpose(K))`, scaling, optional `c * Tanh(scores / c)` for positive
softcap `c`, additive position bias, then
`Where(allowed, scores, -inf)`. This is an example order, not a prescription
for models whose bias or softcap placement differs. GPT-OSS uses scale followed
by an additive causal/window mask and no softcap in the linked implementation.

Boolean masks use `True` for allowed real positions in this composition.
Floating masks/biases use `Add` in the model's specified order; an additive
`-inf` mask should be applied after softcap, not turned into a finite value
by `Tanh`. Compute causal, sliding-window, and padding eligibility using the
**original** query/key positions. For example, with absolute query position
`q`, a causal window of width `w > 0` allows keys `q - w < k <= q`, intersected
with padding validity. Do not shift these positions or recompute causal
alignment as though the key length were `K + 1`. Append the sink only after
all such operations, so it remains visible even when no real key is allowed.

## Recognizable subgraph

For opset 13 or later, the core dataflow is:

```text
S [Hq] -- Reshape [1,Hq,1,1] -- Expand [B,Hq,Q,1] -- S_full
L [B,Hq,Q,K] ---------------------------------------+
                                                   |
Z = Concat(L, S_full, axis=-1)           # [B,Hq,Q,K+1]
P_all = Softmax(Z, axis=-1)              # normalize real + sink
P = Slice(P_all, [0], [-1], [-1], [1])   # retain only real keys
Y = MatMul(P, V)                        # [B,Hq,Q,Dv]
```

`Slice` inputs above are `starts`, `ends`, `axes`, and `steps`, all int64.
The exclusive end `-1` removes exactly the last candidate, including when
`K = 0`. `Concat` itself does **not** broadcast: its non-concatenated
dimensions must match. `Expand` supplies that match. Derive its target shape
from the first three dimensions of `Shape(L)` followed by `[1]`, rather than
hard-coding sequence lengths.

`S` can be a graph input (as in the example), an initializer containing learned
weights, or a zero constant for smooth softmax. A scalar shared sink can be
reshaped to `[1,1,1,1]` before expansion. A vector `[Hq]` must not be used
directly as a last-axis bias; reshape it to put the head axis in position 1.
More general per-row sinks may already have shape `[B,Hq,Q,1]`, but a runtime
specialization for per-head sinks must verify independence from batch/query.

The graph assumes values have already been aligned to query heads. For grouped
query attention (GQA), require `Hq = Hkv * G` with integer `G`. In
`[B,Hkv,K,D]` layout, `Reshape` to `[B,Hkv,1,K,D]`, `Expand` to
`[B,Hkv,G,K,D]`, and `Reshape` to `[B,Hq,K,D]` for both keys and values.
Thus query head `h` uses KV head `floor(h/G)`, while `S[h]` remains distinct.
The same rule with `Hkv = 1` covers multi-query attention. Projections,
positional embeddings, head layout transforms, and KV gathering/updating
precede the core subgraph; they do not add a sink to stored K or V.

### Precision and exceptional inputs

The executable example below accepts float16, float32, or float64 storage.
It casts **final** `L`, `S`, and `V` to float32 for float16/float32 storage,
or retains float64 for float64 storage, before expansion/normalization and
the value contraction. `P` stays in that computation type; only `Y` is cast
back to the storage type. This is a declared choice for this example, not
an implicit accumulation-type guarantee of `Softmax` or `MatMul`.

Export the source model's actual cast boundaries. Casting Q/K before their
contraction can differ from casting already-rounded logits afterward;
casting probabilities down before multiplying V can also differ from this
example. A fusion must preserve these boundaries and rounding behavior under
the runtime's numerical contract, not silently choose another precision.
The example does not model bfloat16 storage or quantized attention.

Assume finite sinks and finite real values. Real logits may be finite or
`-inf` for masked entries; NaNs and positive infinities are outside this
example's numerical contract. With a finite sink, an all-real-masked row
has zero real probabilities and zero output. With `K = 0`, softmax still
has the one sink candidate, `P` has shape `[B,Hq,Q,0]`, and the empty
contraction produces zeros of shape `[B,Hq,Q,Dv]`. A masked value containing
NaN is not made harmless by a zero probability (`0 * NaN` is NaN).

## Why the sink has zero value contribution

In exact arithmetic, append one **zero value vector** to `V` and retain all
of `P_all`. Its product is

$$
[p_0,\ldots,p_{K-1},p_s]\,[V_0,\ldots,V_{K-1},0]^T
= \sum_{j=0}^{K-1}p_j V_j + p_s\,0 = Y.
$$

This proves the denominator-only effect; it does not require an actual
dummy key, value, or cache entry. For `s = 0`, the denominator is
`1 + sum(exp(L))`, often called *smooth softmax*. For nonempty real support,
the real probabilities are ordinary attention probabilities multiplied by
`sum(exp(L)) / (exp(s) + sum(exp(L)))`. Do not renormalize them after slicing:
their sum is generally less than one.

Appending zero K/V to an `Attention` invocation is not the canonical pattern
here: its causal/window masking, cache lengths, and softcap/bias ordering
would need separate verification. The explicit logits composition avoids
making the virtual candidate participate in those operations.

## Runtime recognition and fusion

A runtime can match the dataflow rather than node names: a last-axis `Concat`
of final real logits and one broadcast sink candidate, last-axis `Softmax`,
a `Slice` retaining exactly the original `K` candidates in order, and a
`MatMul` with the corresponding real values. Equivalent positive axes,
constant-folded reshape/expand shapes, and an initializer instead of a sink
input may be recognized after checking their semantics. Match any intervening
casts as part of the precision contract. A narrower kernel may accept only
per-query-head sinks and a particular layout; other valid graphs can continue
to use the unfused operators.

Check that scaling, softcap, and masks affect only the real-logit input.
Check the GQA head mapping, the softmax axis and opset, the exact slice bounds,
and any other consumers of `P_all`, `P`, or `L`. A fusion must preserve all
observable outputs, including real-token debug probabilities with shape
`[B,Hq,Q,K]`; it cannot erase a consumer it does not implement. Actual KV state
and cache lengths remain unchanged. Graph structure alone does not authorize
reordering floating-point operations or reinterpreting a mask.

A stable online-softmax kernel need not materialize `Z` or `P_all`. For each
row, initialize running maximum `m = s`, denominator `d = 1`, and value
numerator vector `a = 0`. For each block of allowed real logits `l_j` and
values `v_j`, use

$$
m' = \max(m,\max_j l_j), \quad r = \exp(m-m'), \quad
d' = r d + \sum_j \exp(l_j-m'), \quad
a' = r a + \sum_j \exp(l_j-m')v_j.
$$

Skip fully masked/empty blocks, or give their entries zero weight. The
finite sink keeps `m` finite, avoiding an all-masked `-inf - (-inf)`.
Return `a / d`. The initial denominator represents the sink and its initial
numerator represents its zero value. Do not initialize with `exp(s)` in
unshifted units: that can overflow. Use the graph's computation type and
honor its cast boundaries. Block reductions can change rounding, so this
is a numerically stable fusion strategy, not a bitwise-equivalence promise.
If real probabilities are requested, compute them from the final `m,d`
without renormalizing over real keys alone.

## Executable example

The example starts at the final real-logit boundary, keeping projection,
mask construction, and cache policy model-specific. It uses only existing
standard-domain operators at opset 13. From the repository root, run
`python docs/docsgen/source/technical/attention_sink_example.py` (use
backslashes for the path in Windows shells). It checks the graph and evaluates
two query heads with different sinks. The accompanying
`tests/python/attention_sink_composition_test.py` exercises this same graph
against an independent scalar oracle, including original-position masks,
softcap ordering, grouped heads, empty keys, and precision choices.

```{literalinclude} attention_sink_example.py
:language: python
```
