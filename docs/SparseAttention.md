<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Portable sparse-attention compositions

Sparse support patterns do not necessarily require a new ONNX operator.
Existing standard operators can express selected-key attention, joint local and
auxiliary attention, and block-sparse support masks. These are **executable
examples and conformance tests**, not a new sparse operator, cache ABI, or promise
of sparse-kernel performance.

The graph builders and numerical checks live in
[`tests/python/sparse_attention_test.py`](../tests/python/sparse_attention_test.py).
After installing ONNX and the test dependencies, run them from the repository root:

```sh
pixi run pytest tests/python/sparse_attention_test.py
```

The builders are test-local examples, not public library APIs. They use standard
opset 25, float32 tensors, finite input values, fixed positive source lengths,
and deliberately fixed shapes to keep the graph transformations visible:
batch 2, query heads 4, KV heads 2, query length 3, main KV length 5, auxiliary
KV length 4, query/key head size 2, and value head size 1.
Query and KV lengths need not match. Each query head `h` uses KV head `h // 2`
(contiguous GQA groups). Each query shares its selection across heads; per-head
selectors are outside these examples.
GQA shares K/V heads, not necessarily selection policies: a per-head selector
cannot be substituted into the shared-selection shape. Such an exporter must
carry its head axis through gathering and masking (or fold query heads into
batch with the corresponding KV-head mapping), preserving each head's list.

## Canonical selected-position composition

The portable boundary is an ordinary tensor of selected source-row positions,
not a new operator signature. A selector produces positions and validity; the
graph gathers selected K/V and their original token coordinates, folds query
positions into batch, calls standard `Attention`, and restores the output layout:

```text
selected positions -> validity + safe positions -> GatherND(K, V, coordinates)
Q -> Transpose/Reshape --------------------------> Attention -> Reshape/Transpose
gathered coordinates + query coordinates --------> boolean mask -----^
```

The selection tensor can be reused by multiple attention layers while each layer
gathers its own K/V. Keeping that tensor as a graph value exposes shared index
reuse without prescribing how selection is computed or how KV is cached.

### Mapping model selectors

[GLM-5.3's configuration](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)
uses DSA and a schedule of full and shared indexer layers. The
[pinned GLM DSA implementation](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L153-L242)
produces int32 top-k row positions; `GlmMoeDsaAttention` in the same file passes
`prev_topk_indices` between layers according to `indexer_types`. An exporter can
wire one selection value to the corresponding gather-and-attend compositions.
Layers still use their own projections, K/V and attention scaling. Indexer
scoring and MLA projections are upstream of this composition, not implemented
by the examples.

[Qwen3.8-Flash-Next's official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
describes it as a preview of the architecture underpinning Qwen4. Its
[pinned QSA implementation](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/qwen4_exp/modular_qwen4_exp.py#L368-L508)
groups each query's **compact visible-position list** in groups of four, selects
complete groups, expands their actual member positions, then appends the visible
partial tail. For visible positions `[0, 2, 3, 5, 6, 8]`, the complete group is
`[0, 2, 3, 5]` and the tail is `[6, 8]`. These are not necessarily contiguous
global blocks: multiplying a selected group ID by four would lose visibility
holes. With the released budget of 2048 tokens and compression ratio four, the
implementation stores int32 `[B, Sq, 2051]` positions with `-1` padding before
building its dense support mask. An exporter can instead retain those selected
row positions for gathering, including the tail, and retain the original
visibility mask. If the K/V tensor is itself compacted, map original token
positions to its row indices and carry original coordinates separately.

The runnable examples use a count-valid prefix, rather than a sentinel-only
interface. For a `-1`-padded selector, preserve `indices >= 0` as slot validity
or supply counts when valid entries form a prefix. Widen int32 positions with
`Cast(to=INT64)` for the existing opset-25 `GatherND`; the
`int32_indices=True` example exercises this route. Apply `Where(valid, index, 0)`
**before** gathering and keep validity in the attention mask. Passing `-1`
directly to `GatherND` would select the last source row, not represent padding.
Neither widening nor safe replacement licenses an out-of-range active index.

These model selectors describe set-valued support. Their unique row positions
map to the examples' multiset composition without changing weights; arbitrary
duplicate indices do not. In particular, a dense scatter-to-boolean-mask
implementation collapses duplicates, whereas repeated gathered candidates
participate repeatedly in softmax.

### Runtime recognition and fusion

The selected-position tensor, its validity and the source K/V remain visible
to an optimizing runtime. It can recognize the shared-index gather, layout
changes, coordinate masking and Attention as one region, retain the original
index tensor and load selected K/V directly from the source. When internal
values have no other consumers, fusion can avoid materializing source
transposes, gathered K/V tensors, folded layout copies, gathered-coordinate and
mask tensors, and local/auxiliary concatenations. Query/output reshapes and
transposes may become layout calculations rather than copies. A fused
implementation can also avoid materializing attention scores/probabilities.

This is an optimization opportunity, not a required execution plan or measured
speed claim. A legal rewrite must preserve index validity, source-row identity,
duplicate multiplicity, head mapping, coordinate masks, scaling, one shared
normalization for combined candidates, and empty-row behavior. It must preserve
other graph consumers of intermediates. Reusing an index tensor across layers
does not imply sharing K/V values or changing the selector. The unfused graph
remains executable using standard operators.

## Selected-only attention

`_selected_attention_model(local_plus_selected=False)` takes ordinary contiguous
main K/V tensors and request-local selected row indices:

- `indices[batch, query, slot]` has capacity 3, with int64 indices.
- `counts[batch, query]` is in `[0, 3]`. Only the prefix of this length is active.
- Active indices must be in `[0, source_length)`. They refer only to that batch
  item's source rows, never to a global/paged cache.
- Inactive slots may contain arbitrary integers. `Where` replaces them with zero
  **before** `GatherND`; their mask remains false. Even an entirely empty selection
  therefore never gathers an invalid index. Source length must be positive;
  capacity remains positive even when every count is zero.
- This graph preserves duplicate active indices as a **multiset**: a repeated
  candidate contributes repeatedly to the softmax denominator and numerator.
  Exporters intending a set-valued sparse support must supply unique indices.
  There is no implicit deduplication or validation of invalid active indices/counts.

K/V transpose to `[B, L, Hkv, D]`; `GatherND(batch_dims=1)` produces
`[B, Q, C, Hkv, D]`. Transpose and reshape fold query positions into batch:
Q becomes `[B*Q, Hq, 1, D]` and gathered K/V become `[B*Q, Hkv, C, D]`.
One ordinary `Attention` processes these per-query candidate lists, using a
boolean mask `[B*Q, 1, 1, C]`. Reshape and transpose restore `[B, Hq, Q, Dv]`.

The caller supplies `query_positions[B,Q]` and `key_positions[B,L]` in the same
original token-coordinate system. The graph gathers source positions along with
K/V, then allows a candidate only if its original position is at most the query
position. `Attention.is_causal` is **zero**: applying a causal triangle to gathered
ordinal positions would be incorrect, especially for reordered candidates and
unequal query/KV lengths. No position offset is inferred from tensor lengths.
For additional source padding, packed-sequence or other visibility restrictions,
gather the corresponding per-query mask at the same row indices and intersect
it with validity and coordinate-based causal/window masks. Selected positions
alone do not remove the need to preserve the model's original attention mask;
for example, top-k can return masked positions when too few valid keys exist.
These small examples supply only count validity and the documented coordinate
restrictions, not a general packed-sequence mask input.
An empty or fully masked selection produces zero under Attention-25 semantics.
Here "empty" means a positive-capacity candidate tensor with all mask entries
false, not a zero-length K/V axis. The tests exercise this path with finite
inputs in both native and function-expanded Attention. They do not establish
behavior for zero source lengths, nonfinite K/V or earlier Attention versions.
An exporter targeting a different contract must handle such cases explicitly.

## Local main plus selected auxiliary candidates

`_selected_attention_model(local_plus_selected=True)` uses a separate auxiliary
K/V source with the same KV-head mapping and head sizes. Here `indices` addresses
auxiliary rows, with `aux_positions[B,4]` in the same original coordinate system.
Main candidates satisfy `0 <= query_position - key_position <= 1`: a causal
window including the current position and its immediate predecessor.
Auxiliary candidates use the count and original-coordinate causal masks, but no
local-window restriction.

The graph concatenates main and selected auxiliary K/V **before one Attention**,
and concatenates their boolean masks correspondingly. Both candidate sets share
one softmax denominator. Two local values of 2 and one auxiliary value of 10,
with zero logits, produce `14/3`, not `2 + 10`. If the auxiliary entry is repeated
twice, the result is 6. An empty auxiliary selection leaves ordinary local
attention; if both sets are fully masked, the result is zero.

Main and auxiliary rows are distinct candidates even if they have the same token
coordinate or equal contents. There is no cross-source identity/deduplication
rule. This small example materializes all main rows and masks to the window;
an exporter could instead construct a compact local candidate list.

## Block-sparse support as a boolean mask

`_block_attention_model()` takes a boolean block layout
`[B, Hkv, ceil(Q/2), ceil(L/2)]`, with block size 2. `Gather` maps KV heads to
query heads and each token ordinal to its block ordinal (`i // 2`, `j // 2`),
including partially occupied final blocks. An entry is allowed only if its block
is present **and** its original key coordinate is no later than its original
query coordinate. The resulting `[B,Hq,Q,L]` mask feeds ordinary Attention.
Block membership uses tensor ordinals; causality uses supplied token coordinates.

This is a dense semantic fallback: both the mask and attention scores may be
quadratic in sequence length. A runtime may recognize/fuse compact gathered
candidate graphs or sparse masks, but these decompositions do not require or
guarantee sparse execution.

## Scope and runtime context

The examples are motivated by ONNX Runtime contrib operators, pinned to
[`526e95a1447a43af1d1ac71c0cd8b1de62b47f5d`](https://github.com/microsoft/onnxruntime/tree/526e95a1447a43af1d1ac71c0cd8b1de62b47f5d):

- [`DynamicSparseAttention`](https://github.com/microsoft/onnxruntime/blob/526e95a1447a43af1d1ac71c0cd8b1de62b47f5d/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L637-L751)
  documents externally selected indices/counts and selected-only versus
  local-plus-selected policies. Its
  [tests](https://github.com/microsoft/onnxruntime/blob/526e95a1447a43af1d1ac71c0cd8b1de62b47f5d/onnxruntime/test/contrib_ops/dynamic_sparse_attention_op_test.cc#L887-L1029)
  exercise GQA, joint normalization and empty selection.
- [`SparseAttention`](https://github.com/microsoft/onnxruntime/blob/526e95a1447a43af1d1ac71c0cd8b1de62b47f5d/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L2286-L2400)
  uses block-CSR metadata; its
  [parity tests](https://github.com/microsoft/onnxruntime/blob/526e95a1447a43af1d1ac71c0cd8b1de62b47f5d/onnxruntime/test/python/transformers/test_sparse_attention.py#L91-L174)
  expand block layouts to boolean dense masks.

These are mathematical compositions, **not** drop-in decompositions of every
contrib input or an ORT parity claim. ORT's DynamicSparseAttention schema requires
unique active indices and `-1` padding; these examples explicitly allow multiset
handling and arbitrary inactive padding instead. Paged block
tables, in-place cache updates, selectors/top-k policies, quantized cache bytes,
fused RoPE, scratch capacities, GPU dispatch and attention sinks are out of scope.
Those runtime concerns do not justify transplanting a bespoke contrib schema into
the portable standard just to express the supported mathematics.

The separate [FlexAttention proposal (#7523)](https://github.com/onnx/onnx/pull/7523)
explores programmable attention. These examples neither implement nor depend on
it, and do not assess that proposal's correctness.

The tests execute the composed models with `ReferenceEvaluator`, both with native
Attention and with its schema function expanded to standard primitive operators.
Models pass the checker and strict shape inference. Independent dense enumeration
and hand-computable cases cover shared normalization, GQA head mapping, unequal
lengths, reordered original coordinates, duplicate multiplicity, empty selection,
arbitrary unused padding, per-batch layouts and partial/fully masked blocks.
Focused cases also execute the int32-to-int64 path with `-1` padding and
noncontiguous original coordinates. The model-selector mappings above are
source-based explanations; these tests do not execute the GLM/Qwen indexers.
The linked ORT source tests are evidence for the design, not executions performed
by this conformance suite.
