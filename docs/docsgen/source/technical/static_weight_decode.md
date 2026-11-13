<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(onnx-detail-static-weight-decode)=

# Fusing static weight decode with computation

Quantized weights can be represented as a decode node followed by a standard
floating-point consumer. This gives exporters a portable graph and backends
a recognizable pattern for kernels that decode weights as they use them.
Fusion avoids writing and reading a full floating-point weight intermediate;
for lookups and expert selection, it can also skip decoding unselected weights.

This guide covers fixed weight shapes and layouts. Batch sizes, token counts,
and selections can still vary at runtime. The examples use opset 29, including
DequantizePacked and GroupedMatMul. Operator calls below are schematic:
attributes are written as named arguments, and letters in weight shapes stand
for literal integer dimensions supplied by the exporter.

## Decode representations

[DequantizePacked](../operators/onnx__DequantizePacked.md) consumes a rank-one
UINT8 payload containing consecutive quantization blocks:

```text
W = DequantizePacked(packed, format="gguf:q4_0", shape=[O,K], output_dtype=FLOAT)
```

The `format`, `shape`, and `output_dtype` attributes describe the fixed weights.
Shape inference reads the logical weight dimensions from `shape`.
For a format with `b` elements and `s` bytes per block, the last dimension
must be divisible by `b`, and the payload length is
`product(shape[:-1]) * (shape[-1] / b) * s`. For Q4_0, `b=32` and `s=18`.
The packed layout fixes how the backend locates blocks and their scales.

[DequantizeLinear](../operators/onnx__DequantizeLinear.md) operates on a typed
quantized tensor that already has the logical weight shape. For example,
an FP8 tensor of shape `[O,K]` with blocks of `B` elements along `K` uses:

```text
W = DequantizeLinear(fp8_weights, scales, axis=1, block_size=B, output_dtype=FLOAT)
```

Here `scales` has shape `[O,ceil(K/B)]`. The positive `block_size` and `axis`
describe the scale layout; the final block can be partial. For per-tensor
scaling, use a scalar scale and omit `block_size`. For per-axis scaling, use
a one-dimensional scale of length equal to the selected axis and omit
`block_size`. The examples omit the optional zero point, giving zero.
An exporter can supply weights and scales as initializers.

The following patterns use FLOAT activations and decoded weights. Other
floating-point output types can be used where supported by both operators;
the consumer's activation, weight, and bias types must agree.

## Dense attention, MLP, and output projections

Query, key, value, attention-output, MLP, and vocabulary-output projections
all use the same weight-decode/matrix-product pattern:

```text
W [K,O] = DequantizePacked(packed, format=..., shape=[K,O])
Y [T,O] = MatMul(X [T,K], W)
```

[MatMul](../operators/onnx__MatMul.md) also preserves leading activation
dimensions: `X [batch,tokens,K]` gives `Y [batch,tokens,O]`.
These activation dimensions may be symbolic. With packed `[K,O]` weights,
`O` must be block-aligned. For typed FP8 `[K,O]` weights, blocks along the
contraction dimension instead use `DequantizeLinear(axis=0, block_size=B)`
with scales `[ceil(K/B),O]`.

Weights stored as output rows fit [Gemm](../operators/onnx__Gemm.md) directly:

```text
W [O,K] = DequantizePacked(packed, format=..., shape=[O,K])
Y [T,O] = Gemm(X [T,K], W, bias [O], transB=1)
```

This uses the default `alpha=1` and `beta=1`; `K` is block-aligned.
The FP8 alternative decodes `[O,K]` with `axis=1`, `block_size=B`, and
scales `[O,ceil(K/B)]`. For a MatMul consumer with this storage layout,
place a constant-permutation `Transpose(perm=[1,0])` between decode and MatMul.
A backend can fold that layout change into its weight access.
Fused projection kernels decode weight tiles while computing the product,
reusing each tile across activation rows as appropriate.

## Convolutions

[Conv](../operators/onnx__Conv.md) consumes weights in
`[Cout,Cin/group,kH,kW]` order for a two-dimensional convolution:

```text
W [Cout,Cin/group,kH,kW] = DequantizeLinear(fp8_filters, scales, axis=1, block_size=B)
Y = Conv(X [batch,Cin,H,W], W, bias [Cout], group=group)
```

For these FP8 blocks along the input-channel axis, scales have shape
`[Cout,ceil((Cin/group)/B),kH,kW]`. A per-output-channel scale instead has
shape `[Cout]` with `axis=0` and `block_size` omitted.
The convolution's strides, pads, dilations, and groups retain their usual
meaning. A fused kernel decodes filter tiles as it traverses channels and
spatial positions.

DequantizePacked can also produce this filter shape directly when `kW`
is block-aligned. For filters with a smaller innermost dimension, an exporter
can pack flattened output-channel rows:

```text
rows [Cout,R] = DequantizePacked(packed, format=..., shape=[Cout,R])
W = Reshape(rows, filter_shape)
Y = Conv(X, W, bias, group=group)
```

Here `R=(Cin/group)*kH*kW` is block-aligned, and `filter_shape` is a constant
INT64 tensor containing `[Cout,Cin/group,kH,kW]`. The reshape preserves element
order. This representation applies when the payload is stored in that
flattened layout; it does not change the block-alignment rules.

## Embeddings and learned lookup tables

[Gather](../operators/onnx__Gather.md) expresses an embedding lookup:

```text
table [V,H] = DequantizePacked(packed_table, format=..., shape=[V,H])
embeddings [N,H] = Gather(table, token_ids [N], axis=0)
```

`H` is block-aligned and each row consists of independently decodable blocks.
The table shape is fixed while `N` may be symbolic and token IDs vary at
runtime. The same pattern covers learned lookup tables; higher-rank IDs
produce their index shape followed by `[H]`.

For typed FP8 tables, use `DequantizeLinear(axis=1, block_size=B)` with
scales `[V,ceil(H/B)]`, or per-tensor scaling.
A fused lookup reads and decodes the selected rows, preserving Gather's
index order, repeated indices, and negative-index handling.

## Selected-expert projections

[GroupedMatMul](../operators/onnx__GroupedMatMul.md) selects weight matrices
for each input row:

```text
weights [E,O,K] = DequantizePacked(packed_experts, format=..., shape=[E,O,K])
Y [M,S,O] = GroupedMatMul(X [M,K], weights, expert_ids [M,S], transB=1)
```

The integer `expert_ids` values lie in `[0,E)`. `E`, `O`, and `K` are fixed
weight dimensions; `M` and `S` may be symbolic. Each expert contains
independently decodable rows with block-aligned `K`. For FP8 experts, use
`DequantizeLinear(axis=2, block_size=B)` with scales `[E,O,ceil(K/B)]`.

A fused kernel computes selected expert projections directly from their
packed or FP8 storage. It can reuse weights across rows selecting the same
expert and avoid expanding the full floating-point expert bank.
This represents one projection within a mixture-of-experts layer.
The optional fourth GroupedMatMul input is a per-expert bias `[E,O]`,
selected using the same IDs.

## Bias, activation, and residual epilogues

A backend can extend a matched decode/compute pattern through pointwise
consumers to reduce intermediate activation traffic:

```text
W = DequantizePacked(packed, format=..., shape=[K,O])
P [T,O] = MatMul(X [T,K], W)
A [T,O] = Add(P, bias [O])
H [T,O] = Relu(A)
Y [T,O] = Add(H, residual [T,O])
```

Gemm, Conv, and GroupedMatMul also accept bias directly in the layouts shown
above. Match the graph's actual activation, broadcast rules, and operation
order when extending an epilogue. Other pointwise activations can be handled
in the same way when supported by the backend.

## Matching and execution

Start at the compute or lookup operator and trace its weight input to the
decode node. Read the fixed format, dimensions, scale layout, and consumer
attributes to select a kernel. Constant-shape Reshape and
constant-permutation Transpose nodes can be included in the match by
mapping consumer coordinates back to decoded weight coordinates.
Keeping this path explicit makes it recognizable without introducing a
dedicated fused ONNX operator.

For dense products and convolutions, decode tiles as computation uses them.
For Gather and GroupedMatMul, decode the rows or experts selected at runtime.
This combines static weight metadata with runtime-dependent work rather
than requiring a runtime weight-shape input. When a decoded value is shared,
preserve all consumers and graph outputs, either with shared execution or
separate fusion decisions.

Fusion is an optional backend optimization of the floating-point graph.
It preserves the decoder's arithmetic and output conversion and the
consumer's semantics, including epilogue order. In particular,
DequantizePacked specifies FLOAT decode arithmetic followed by conversion
to `output_dtype`; DequantizeLinear specifies multiplication precision
through its output type. A kernel using native low-precision arithmetic
must establish equivalence to the represented decode/compute sequence.
Without fusion, the graph executes with the decoded floating-point weights
as an intermediate.
