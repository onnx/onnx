// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/schema.h"

#include <cstdint>
#include <string>

#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

namespace {

constexpr const char* kKVCacheType = "opaque(org.onnx.experimental.state,KVCache)";
constexpr const char* kLinearAttentionStateType =
    "opaque(org.onnx.experimental.state,LinearAttentionState)";
constexpr const char* kCausalConvStateType =
    "opaque(org.onnx.experimental.state,CausalConvState)";

void SetOpaqueOutputType(InferenceContext& ctx, size_t output_index, const char* name) {
  auto* opaque_type = ctx.getOutputType(output_index)->mutable_opaque_type();
  opaque_type->set_domain(ONNX_EXPERIMENTAL_STATE_DOMAIN);
  opaque_type->set_name(name);
}

void ValidateLossyStateAttribute(InferenceContext& ctx) {
  const auto allow_lossy_state = getAttribute(ctx, "allow_lossy_state", int64_t{0});
  if (allow_lossy_state != 0 && allow_lossy_state != 1) {
    fail_shape_inference("allow_lossy_state must be 0 or 1, got ", allow_lossy_state, ".");
  }
}

void ValidateSequenceLens(InferenceContext& ctx, size_t input_index, const Dim& batch) {
  if (!ctx.hasInput(input_index)) {
    return;
  }
  Dim lengths_batch;
  ctx.unifyInputShape(input_index, {lengths_batch});
  unifyDim(batch, lengths_batch);
}

void AttentionWithCacheInference(InferenceContext& ctx) {
  ValidateLossyStateAttribute(ctx);
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  SetOpaqueOutputType(ctx, 1, "KVCache");

  Dim batch, query_heads, query_length, query_head_size;
  if (hasInputShape(ctx, 0)) {
    const auto& query_shape = getInputShape(ctx, 0);
    if (query_shape.dim_size() == 4) {
      ctx.unifyInputShape(0, {batch, query_heads, query_length, query_head_size});
      Dim value_batch, value_heads, value_length, value_head_size;
      ctx.unifyInputShape(2, {value_batch, value_heads, value_length, value_head_size});
      unifyDim(batch, value_batch);
      updateOutputShape(ctx, 0, {batch, query_heads, query_length, value_head_size});
      if (ctx.hasOutput(2)) {
        propagateElemTypeFromInputToOutput(ctx, 0, 2);
        updateOutputShape(ctx, 2, {batch, query_heads, query_length, Dim()});
      }
    } else if (query_shape.dim_size() == 3) {
      Dim query_hidden;
      ctx.unifyInputShape(0, {batch, query_length, query_hidden});
      updateOutputShape(ctx, 0, {batch, query_length, Dim()});
      if (ctx.hasOutput(2)) {
        propagateElemTypeFromInputToOutput(ctx, 0, 2);
        updateOutputShape(ctx, 2, {batch, Dim(), query_length, Dim()});
      }
    } else {
      fail_shape_inference("Q must have rank 3 or 4, got rank ", query_shape.dim_size(), ".");
    }
  }

  for (const size_t position_input : {size_t{5}, size_t{6}}) {
    if (ctx.hasInput(position_input)) {
      Dim position_batch, position_length;
      ctx.unifyInputShape(position_input, {position_batch, position_length});
      unifyDim(batch, position_batch);
    }
  }
  ValidateSequenceLens(ctx, 7, batch);
}

void LinearAttentionWithStateInference(InferenceContext& ctx) {
  ValidateLossyStateAttribute(ctx);
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  SetOpaqueOutputType(ctx, 1, "LinearAttentionState");

  Dim batch, sequence_length, query_hidden;
  ctx.unifyInputShape(0, {batch, sequence_length, query_hidden});
  updateOutputShape(ctx, 0, {batch, sequence_length, Dim()});
  ValidateSequenceLens(ctx, 6, batch);
}

void CausalConvWithStateInference(InferenceContext& ctx) {
  ValidateLossyStateAttribute(ctx);
  const auto dilation = getAttribute(ctx, "dilation", int64_t{1});
  if (dilation < 1) {
    fail_shape_inference("dilation must be at least 1, got ", dilation, ".");
  }
  propagateShapeAndTypeFromFirstInput(ctx);
  SetOpaqueOutputType(ctx, 1, "CausalConvState");

  Dim batch, channels, length;
  ctx.unifyInputShape(0, {batch, channels, length});
  Dim weight_channels, one, kernel;
  one.set_dim_value(1);
  ctx.unifyInputShape(1, {weight_channels, one, kernel});
  unifyDim(channels, weight_channels);
  if (ctx.hasInput(2)) {
    Dim bias_channels;
    ctx.unifyInputShape(2, {bias_channels});
    unifyDim(channels, bias_channels);
  }
  ValidateSequenceLens(ctx, 4, batch);
}

constexpr const char* kCommonStateSemantics = R"DOC(
This operator is an experimental proposal in the custom
`org.onnx.experimental.state` domain. Its presence does not indicate acceptance
into the ONNX standard.

The state input and output have a nominal Opaque type specific to this
operator family. The input is a logical snapshot and is not mutated. The output
is a new logical snapshot, even when an implementation physically aliases
storage or uses copy-on-write. Implementations may choose dense, segmented,
paged, compressed, device-specific, or other physical representations outside
the graph, provided the logical behavior is preserved. Consequently, one model
graph can run under different state-storage policies.

When `allow_lossy_state` is zero, the physical representation must preserve the
logical state values according to the operator's tensor element types. When it
is one, the model explicitly permits implementation-selected lossy state
storage. This permission does not select or expose a physical representation.
)DOC";

std::string WithCommonStateSemantics(const char* doc) {
  return std::string(kCommonStateSemantics) + "\n" + doc;
}

} // namespace

ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA(
    AttentionWithCache,
    1,
    OpSchema()
        .SetDoc(WithCommonStateSemantics(R"DOC(
Computes the same scaled dot-product attention math as `ai.onnx.Attention`,
while carrying the key/value history in an Opaque `KVCache`.

The logical cache is, for each batch element, an ordered sequence of key/value
records and their absolute positions. An omitted `cache` is an empty sequence.
`K` and `V` append records to that sequence; `present_cache` is the resulting
snapshot. A dense contiguous cache and a paged cache are equivalent physical
representations of this same logical value.

`q_positions` and `kv_positions`, when present, have shapes `(B, Sq)` and
`(B, Skv)` and identify absolute positions. When absent, positions are inferred
as a contiguous append after the cache. For ragged batches, `nonpad_kv_seqlen`
has shape `(B)` and gives the valid prefix length of the current K/V chunk;
records outside that prefix are not appended. Causal and window masks compare
absolute positions. `attn_mask`, when present, is additionally applied over
the logical key order after the cache and current valid records are combined.
)DOC"))
        .Attr("is_causal", "Apply causal masking using absolute positions.", AttributeProto::INT, int64_t{0})
        .Attr("scale", "Scaling factor applied to Q*K^T. The default is 1/sqrt(head_size).", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("q_num_heads", "Number of query heads for rank-3 inputs.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("kv_num_heads", "Number of key/value heads for rank-3 inputs.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("softmax_precision", "Optional softmax computation element type.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("softcap", "Optional positive soft cap for logits; zero disables it.", AttributeProto::FLOAT, 0.0f)
        .Attr("left_window_size", "Maximum relative distance to the left; -1 is unbounded.", AttributeProto::INT, int64_t{-1})
        .Attr("right_window_size", "Maximum relative distance to the right; -1 is unbounded.", AttributeProto::INT, int64_t{-1})
        .Attr("allow_lossy_state", "Whether implementation-selected lossy cache storage is permitted.", AttributeProto::INT, int64_t{0})
        .Input(0, "Q", "Rank-3 packed or rank-4 query tensor.", "T1")
        .Input(1, "K", "Rank-3 packed or rank-4 key tensor for new records.", "T1")
        .Input(2, "V", "Rank-3 packed or rank-4 value tensor for new records.", "T2")
        .Input(3, "attn_mask", "Optional boolean or additive attention mask.", "U", OpSchema::Optional)
        .Input(4, "cache", "Optional prior logical cache snapshot. Omission initializes an empty cache.", "S", OpSchema::Optional)
        .Input(5, "q_positions", "Optional absolute query positions with shape (B, Sq).", "tensor(int64)", OpSchema::Optional)
        .Input(6, "kv_positions", "Optional absolute positions for current K/V with shape (B, Skv).", "tensor(int64)", OpSchema::Optional)
        .Input(7, "nonpad_kv_seqlen", "Optional valid current K/V prefix lengths with shape (B).", "tensor(int64)", OpSchema::Optional)
        .Output(0, "Y", "Attention output, following ai.onnx.Attention layout rules.", "T1")
        .Output(1, "present_cache", "Updated logical cache snapshot.", "S", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(2, "qk_matmul_output", "Optional logical QK score tensor before masking and softmax.", "T1", OpSchema::Optional)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q, K, Y, and score types.")
        .TypeConstraint("T2", OpSchema::all_float_types_ir4(), "Constrain V types.")
        .TypeConstraint("U", OpSchema::all_non_complex_numeric_types_plus_bool_ir4(), "Constrain mask types.")
        .TypeConstraint("S", {kKVCacheType}, "Nominal opaque KV cache type.")
        .TypeAndShapeInferenceFunction(AttentionWithCacheInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic));

ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA(
    LinearAttentionWithState,
    1,
    OpSchema()
        .SetDoc(WithCommonStateSemantics(R"DOC(
Computes the recurrent math of `ai.onnx.LinearAttention`, while carrying the
recurrent accumulator in an Opaque `LinearAttentionState`. The logical state is
the accumulator matrix required by the selected `update_rule`; it is not a KV
sequence and need not use a paged-KV representation.

An omitted `state` is the all-zero accumulator. Tokens are processed in
sequence order using the `linear`, `gated`, `delta`, or `gated_delta`
recurrence defined by `ai.onnx.LinearAttention`. `sequence_lens`, when present,
has shape `(B)` and selects a valid prefix per batch element. Positions outside
that prefix produce zero output and do not update the state.
)DOC"))
        .Attr("update_rule", "One of linear, gated, delta, or gated_delta.", AttributeProto::STRING, std::string("gated_delta"))
        .Attr("scale", "Output scale; zero derives 1/sqrt(d_k).", AttributeProto::FLOAT, 0.0f)
        .Attr("q_num_heads", "Number of query heads.", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of compact key heads.", AttributeProto::INT)
        .Attr("v_num_heads", "Value/state head count; zero uses kv_num_heads.", AttributeProto::INT, int64_t{0})
        .Attr("chunk_size", "Implementation hint that does not change results.", AttributeProto::INT, int64_t{64})
        .Attr("allow_lossy_state", "Whether implementation-selected lossy accumulator storage is permitted.", AttributeProto::INT, int64_t{0})
        .Input(0, "query", "Packed query tensor with shape (B, T, Hq*d_k).", "T")
        .Input(1, "key", "Packed key tensor with shape (B, T, Hkv*d_k).", "T")
        .Input(2, "value", "Packed value tensor with shape (B, T, Hv*d_v).", "T")
        .Input(3, "state", "Optional prior logical accumulator. Omission initializes zeros.", "S", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
        .Input(4, "decay", "Optional decay gate required by gated update rules.", "T", OpSchema::Optional)
        .Input(5, "beta", "Optional update rate required by delta update rules.", "T", OpSchema::Optional)
        .Input(6, "sequence_lens", "Optional valid prefix lengths with shape (B).", "tensor(int64)", OpSchema::Optional)
        .Output(0, "output", "Packed recurrent attention output with shape (B, T, Ho*d_v).", "T")
        .Output(1, "present_state", "Updated logical recurrent accumulator snapshot.", "S", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"}, "Constrain activation types.")
        .TypeConstraint("S", {kLinearAttentionStateType}, "Nominal opaque linear-attention state type.")
        .TypeAndShapeInferenceFunction(LinearAttentionWithStateInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic));

ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA(
    CausalConvWithState,
    1,
    OpSchema()
        .SetDoc(WithCommonStateSemantics(R"DOC(
Computes the same depthwise causal one-dimensional convolution as
`ai.onnx.CausalConvWithState`, while carrying bounded history in an Opaque
`CausalConvState`. The logical state is the last `(K - 1) * dilation` raw input
samples per batch and channel in chronological order. It is bounded history,
not a KV sequence, and implementations may use ring buffers or other suitable
representations.

An omitted `state` initializes that history to zero. `sequence_lens`, when
present, has shape `(B)` and selects a valid prefix per batch element.
Positions outside the prefix produce zero output and do not advance history.
)DOC"))
        .Attr("activation", "One of none, silu, or swish.", AttributeProto::STRING, std::string("none"))
        .Attr("dilation", "Spacing between kernel samples; must be at least 1.", AttributeProto::INT, int64_t{1})
        .Attr("allow_lossy_state", "Whether implementation-selected lossy history storage is permitted.", AttributeProto::INT, int64_t{0})
        .Input(0, "input", "Input tensor with shape (B, C, L).", "T")
        .Input(1, "weight", "Depthwise kernel with shape (C, 1, K).", "T")
        .Input(2, "bias", "Optional per-channel bias with shape (C).", "T", OpSchema::Optional)
        .Input(3, "state", "Optional prior logical history. Omission initializes zeros.", "S", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
        .Input(4, "sequence_lens", "Optional valid prefix lengths with shape (B).", "tensor(int64)", OpSchema::Optional)
        .Output(0, "output", "Convolution output with shape (B, C, L).", "T")
        .Output(1, "present_state", "Updated bounded-history snapshot.", "S", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"}, "Constrain tensor types.")
        .TypeConstraint("S", {kCausalConvStateType}, "Nominal opaque causal-convolution state type.")
        .TypeAndShapeInferenceFunction(CausalConvWithStateInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic));

} // namespace ONNX_NAMESPACE
