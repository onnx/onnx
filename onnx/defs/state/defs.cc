// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

namespace {

constexpr const char* kKVCacheType = "opaque(org.onnx.experimental.state,KVCache)";
constexpr const char* kLinearAttentionStateType = "opaque(org.onnx.experimental.state,LinearAttentionState)";
constexpr const char* kCausalConvStateType = "opaque(org.onnx.experimental.state,CausalConvState)";

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

std::optional<std::vector<int64_t>> ReadIntegerInitializer(InferenceContext& ctx, size_t input_index) {
  const auto* initializer = ctx.getInputData(input_index);
  if (initializer == nullptr) {
    return std::nullopt;
  }
  std::vector<int64_t> values;
  if (initializer->data_type() == TensorProto::INT64) {
    values = ParseData<int64_t>(initializer);
  } else if (initializer->data_type() == TensorProto::INT32) {
    const auto values32 = ParseData<int32_t>(initializer);
    values.assign(values32.begin(), values32.end());
  } else {
    fail_shape_inference("Input ", input_index, " must have type int32 or int64.");
  }
  return values;
}

std::optional<std::vector<int64_t>>
ValidateOffsetsInitializer(InferenceContext& ctx, size_t input_index, const Dim& total, const char* name) {
  const auto values = ReadIntegerInitializer(ctx, input_index);
  if (!values.has_value()) {
    return std::nullopt;
  }
  if (values->empty() || values->front() != 0) {
    fail_shape_inference(name, " must be nonempty and start with 0.");
  }
  for (size_t i = 1; i < values->size(); ++i) {
    if ((*values)[i] < (*values)[i - 1]) {
      fail_shape_inference(name, " must be nondecreasing.");
    }
  }
  if (total.has_dim_value() && values->back() != total.dim_value()) {
    fail_shape_inference(
        name, " must end at the packed token count ", total.dim_value(), ", got ", values->back(), ".");
  }
  return values;
}

void ValidateIndexInitializer(
    InferenceContext& ctx,
    size_t input_index,
    const Dim& upper_bound,
    const char* name,
    bool strictly_increasing) {
  const auto values = ReadIntegerInitializer(ctx, input_index);
  if (!values.has_value()) {
    return;
  }
  int64_t previous = -1;
  for (const auto value : *values) {
    if (value < 0 || (upper_bound.has_dim_value() && value >= upper_bound.dim_value())) {
      fail_shape_inference(name, " contains out-of-range index ", value, ".");
    }
    if (strictly_increasing && value <= previous) {
      fail_shape_inference(name, " must contain unique indices in increasing order.");
    }
    previous = value;
  }
}

int64_t PositiveAttribute(InferenceContext& ctx, const char* name, int64_t default_value) {
  const auto value = getAttribute(ctx, name, default_value);
  if (value <= 0) {
    fail_shape_inference(name, " must be positive, got ", value, ".");
  }
  return value;
}

void AttentionWithCacheInference(InferenceContext& ctx) {
  ValidateLossyStateAttribute(ctx);
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  SetOpaqueOutputType(ctx, 1, "KVCache");

  const auto query_block_size = PositiveAttribute(ctx, "visibility_q_block_size", 1);
  const auto key_block_size = PositiveAttribute(ctx, "visibility_k_block_size", 64);
  const auto visibility_num_heads = PositiveAttribute(ctx, "visibility_num_heads", 1);
  if (key_block_size > std::numeric_limits<int64_t>::max() - 63) {
    fail_shape_inference("visibility_k_block_size is too large.");
  }
  const auto mask_words = (key_block_size + 63) / 64;

  Dim batch, query_heads, query_length, query_head_size;
  if (hasInputShape(ctx, 0)) {
    const auto& query_shape = getInputShape(ctx, 0);
    if (query_shape.dim_size() == 2) {
      if (!ctx.hasInput(8) || !ctx.hasInput(9)) {
        fail_shape_inference("Packed rank-2 Q, K, and V require q_sequence_offsets and kv_sequence_offsets.");
      }
      if (ctx.hasInput(3) || ctx.hasInput(7)) {
        fail_shape_inference("Packed mode does not accept attn_mask or nonpad_kv_seqlen.");
      }
      if (ctx.hasOutput(2)) {
        fail_shape_inference("Packed mode does not define the dense qk_matmul_output.");
      }

      const auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
      const auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
      if (q_num_heads_attr == nullptr || kv_num_heads_attr == nullptr || q_num_heads_attr->i() <= 0 ||
          kv_num_heads_attr->i() <= 0) {
        fail_shape_inference("Packed mode requires positive q_num_heads and kv_num_heads attributes.");
      }
      const auto q_num_heads = q_num_heads_attr->i();
      const auto kv_num_heads = kv_num_heads_attr->i();
      if (visibility_num_heads != 1 && visibility_num_heads != q_num_heads) {
        fail_shape_inference("visibility_num_heads must be 1 or q_num_heads in packed mode.");
      }

      Dim total_query, query_hidden, total_new_kv, key_hidden, value_total_new_kv, value_hidden;
      ctx.unifyInputShape(0, {total_query, query_hidden});
      ctx.unifyInputShape(1, {total_new_kv, key_hidden});
      ctx.unifyInputShape(2, {value_total_new_kv, value_hidden});
      unifyDim(total_new_kv, value_total_new_kv);

      Dim output_hidden;
      if (query_hidden.has_dim_value() && key_hidden.has_dim_value()) {
        if (query_hidden.dim_value() % q_num_heads != 0 || key_hidden.dim_value() % kv_num_heads != 0) {
          fail_shape_inference("Packed Q and K hidden sizes must be divisible by their head counts.");
        }
        if (query_hidden.dim_value() / q_num_heads != key_hidden.dim_value() / kv_num_heads) {
          fail_shape_inference("Packed Q and K head sizes must match.");
        }
      }
      if (value_hidden.has_dim_value()) {
        if (value_hidden.dim_value() % kv_num_heads != 0) {
          fail_shape_inference("Packed V hidden size must be divisible by kv_num_heads.");
        }
        const auto value_head_size = value_hidden.dim_value() / kv_num_heads;
        if (value_head_size > 0 && q_num_heads > std::numeric_limits<int64_t>::max() / value_head_size) {
          fail_shape_inference("Packed output hidden size exceeds int64.");
        }
        output_hidden.set_dim_value(q_num_heads * value_head_size);
      }
      updateOutputShape(ctx, 0, {total_query, output_hidden});

      Dim offsets_length;
      ctx.unifyInputShape(8, {offsets_length});
      Dim kv_offsets_length;
      ctx.unifyInputShape(9, {kv_offsets_length});
      unifyDim(offsets_length, kv_offsets_length);
      const auto q_offsets = ValidateOffsetsInitializer(ctx, 8, total_query, "q_sequence_offsets");
      ValidateOffsetsInitializer(ctx, 9, total_new_kv, "kv_sequence_offsets");

      if (ctx.hasInput(5)) {
        Dim query_positions;
        ctx.unifyInputShape(5, {query_positions});
        unifyDim(total_query, query_positions);
      }
      if (ctx.hasInput(6)) {
        Dim kv_positions;
        ctx.unifyInputShape(6, {kv_positions});
        unifyDim(total_new_kv, kv_positions);
      }

      const bool has_visibility_offsets = ctx.hasInput(10);
      const bool has_visibility_blocks = ctx.hasInput(11);
      const bool has_partial_indices = ctx.hasInput(12);
      const bool has_partial_mask = ctx.hasInput(13);
      const bool has_attention_bias = ctx.hasInput(14);
      if (has_visibility_offsets != has_visibility_blocks) {
        fail_shape_inference("visibility_row_offsets and visibility_key_blocks must be provided together.");
      }
      if (has_partial_indices != has_partial_mask) {
        fail_shape_inference(
            "visibility_partial_block_indices and visibility_partial_block_mask must be provided together.");
      }
      if ((has_partial_indices || has_attention_bias) && !has_visibility_blocks) {
        fail_shape_inference("Partial block masks and attn_bias require compact visibility inputs.");
      }
      if (has_visibility_blocks) {
        Dim visibility_rows;
        ctx.unifyInputShape(10, {visibility_rows});
        Dim active_blocks;
        ctx.unifyInputShape(11, {active_blocks});
        ValidateOffsetsInitializer(ctx, 10, active_blocks, "visibility_row_offsets");
        ValidateIndexInitializer(ctx, 11, Dim(), "visibility_key_blocks", false);

        if (q_offsets.has_value()) {
          int64_t expected_rows = 1;
          for (size_t i = 1; i < q_offsets->size(); ++i) {
            const auto sequence_length = (*q_offsets)[i] - (*q_offsets)[i - 1];
            const auto query_blocks = (sequence_length + query_block_size - 1) / query_block_size;
            if (query_blocks > (std::numeric_limits<int64_t>::max() - expected_rows) / visibility_num_heads) {
              fail_shape_inference("Compact visibility row count exceeds int64.");
            }
            expected_rows += query_blocks * visibility_num_heads;
          }
          Dim expected_visibility_rows;
          expected_visibility_rows.set_dim_value(expected_rows);
          unifyDim(visibility_rows, expected_visibility_rows);
        }

        if (has_partial_indices) {
          Dim partial_blocks;
          ctx.unifyInputShape(12, {partial_blocks});
          Dim mask_partial_blocks, mask_query_block, mask_word_count;
          ctx.unifyInputShape(13, {mask_partial_blocks, mask_query_block, mask_word_count});
          unifyDim(partial_blocks, mask_partial_blocks);
          Dim expected_query_block, expected_mask_words;
          expected_query_block.set_dim_value(query_block_size);
          expected_mask_words.set_dim_value(mask_words);
          unifyDim(mask_query_block, expected_query_block);
          unifyDim(mask_word_count, expected_mask_words);
          ValidateIndexInitializer(ctx, 12, active_blocks, "visibility_partial_block_indices", true);
        }
        if (has_attention_bias) {
          Dim bias_active_blocks, bias_query_block, bias_key_block;
          ctx.unifyInputShape(14, {bias_active_blocks, bias_query_block, bias_key_block});
          unifyDim(active_blocks, bias_active_blocks);
          Dim expected_query_block, expected_key_block;
          expected_query_block.set_dim_value(query_block_size);
          expected_key_block.set_dim_value(key_block_size);
          unifyDim(bias_query_block, expected_query_block);
          unifyDim(bias_key_block, expected_key_block);
        }
      }
    } else if (query_shape.dim_size() == 4) {
      for (size_t packed_input = 8; packed_input < 15; ++packed_input) {
        if (ctx.hasInput(packed_input)) {
          fail_shape_inference("Packed offsets and compact visibility inputs require rank-2 Q, K, and V.");
        }
      }
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
      for (size_t packed_input = 8; packed_input < 15; ++packed_input) {
        if (ctx.hasInput(packed_input)) {
          fail_shape_inference("Packed offsets and compact visibility inputs require rank-2 Q, K, and V.");
        }
      }
      Dim query_hidden;
      ctx.unifyInputShape(0, {batch, query_length, query_hidden});
      updateOutputShape(ctx, 0, {batch, query_length, Dim()});
      if (ctx.hasOutput(2)) {
        propagateElemTypeFromInputToOutput(ctx, 0, 2);
        updateOutputShape(ctx, 2, {batch, Dim(), query_length, Dim()});
      }
    } else {
      fail_shape_inference("Q must have rank 2, 3, or 4, got rank ", query_shape.dim_size(), ".");
    }
  }

  if (hasInputShape(ctx, 0) && getInputShape(ctx, 0).dim_size() != 2) {
    for (const size_t position_input : {size_t{5}, size_t{6}}) {
      if (ctx.hasInput(position_input)) {
        Dim position_batch, position_length;
        ctx.unifyInputShape(position_input, {position_batch, position_length});
        unifyDim(batch, position_batch);
      }
    }
    ValidateSequenceLens(ctx, 7, batch);
  }
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

Packed mode uses rank-2 token-major Q, K, and V with independent cumulative
`q_sequence_offsets` and `kv_sequence_offsets`. Each offset tensor has shape
`(B + 1)`, starts at zero, is nondecreasing, and ends at its packed token count;
repeated offsets represent empty sequences. Packed Y preserves Q token order.
Packed positions are rank-1. `attn_mask`, `nonpad_kv_seqlen`, and
`qk_matmul_output` are available only in padded mode.

Compact packed visibility is block CSR over logical keys, never cache pages.
Rows enumerate batch, then visibility head, then local query block.
`visibility_num_heads=1` broadcasts one head-shared pattern; setting it to
`q_num_heads` represents exact head-specific visibility. Active key-block
indices are local to each batch element's logical sequence (prior cache,
followed by current K/V). Active blocks are fully visible unless listed in
`visibility_partial_block_indices`; the corresponding uint64 mask row stores
exact visibility bits. Out-of-range bits in terminal blocks are ignored.
Blocks not listed by a CSR row are invisible. Thus arbitrary visibility,
partial blocks, and fully masked query rows are lossless without a dense mask.
Fully masked rows produce zero Y. Optional `attn_bias` is additive and aligned
to active logical blocks separately from visibility. Compact visibility is
intersected with causal and window rules and may be shared by compatible
layers.
)DOC"))
        .Attr("is_causal", "Apply causal masking using absolute positions.", AttributeProto::INT, int64_t{0})
        .Attr(
            "scale",
            "Scaling factor applied to Q*K^T. The default is 1/sqrt(head_size).",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr("q_num_heads", "Number of query heads for rank-3 inputs.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("kv_num_heads", "Number of key/value heads for rank-3 inputs.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("softmax_precision", "Optional softmax computation element type.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("softcap", "Optional positive soft cap for logits; zero disables it.", AttributeProto::FLOAT, 0.0f)
        .Attr(
            "left_window_size",
            "Maximum relative distance to the left; -1 is unbounded.",
            AttributeProto::INT,
            int64_t{-1})
        .Attr(
            "right_window_size",
            "Maximum relative distance to the right; -1 is unbounded.",
            AttributeProto::INT,
            int64_t{-1})
        .Attr(
            "allow_lossy_state",
            "Whether implementation-selected lossy cache storage is permitted.",
            AttributeProto::INT,
            int64_t{0})
        .Attr("visibility_q_block_size", "Packed visibility query block size.", AttributeProto::INT, int64_t{1})
        .Attr("visibility_k_block_size", "Packed visibility logical key block size.", AttributeProto::INT, int64_t{64})
        .Attr(
            "visibility_num_heads",
            "Packed visibility head count: 1 or q_num_heads.",
            AttributeProto::INT,
            int64_t{1})
        .Input(0, "Q", "Rank-2 packed token-major, rank-3 padded packed-hidden, or rank-4 query tensor.", "T1")
        .Input(
            1,
            "K",
            "Rank-2 packed token-major, rank-3 padded packed-hidden, or rank-4 key tensor for new records.",
            "T1")
        .Input(
            2,
            "V",
            "Rank-2 packed token-major, rank-3 padded packed-hidden, or rank-4 value tensor for new records.",
            "T2")
        .Input(3, "attn_mask", "Optional padded-mode boolean or additive attention mask.", "U", OpSchema::Optional)
        .Input(
            4,
            "cache",
            "Optional prior logical cache snapshot. Omission initializes an empty cache.",
            "S",
            OpSchema::Optional)
        .Input(
            5,
            "q_positions",
            "Optional absolute query positions with padded shape (B, Sq) or packed shape (total_q).",
            "P",
            OpSchema::Optional)
        .Input(
            6,
            "kv_positions",
            "Optional absolute positions for current K/V with padded shape (B, Skv) or packed shape (total_new_kv).",
            "P",
            OpSchema::Optional)
        .Input(
            7,
            "nonpad_kv_seqlen",
            "Optional valid current K/V prefix lengths with shape (B).",
            "tensor(int64)",
            OpSchema::Optional)
        .Input(
            8,
            "q_sequence_offsets",
            "Packed-mode cumulative query offsets with shape (B + 1).",
            "I",
            OpSchema::Optional)
        .Input(
            9,
            "kv_sequence_offsets",
            "Packed-mode cumulative new-K/V offsets with shape (B + 1).",
            "I",
            OpSchema::Optional)
        .Input(10, "visibility_row_offsets", "Optional packed block-CSR row offsets.", "I", OpSchema::Optional)
        .Input(11, "visibility_key_blocks", "Optional active logical key-block indices.", "I", OpSchema::Optional)
        .Input(
            12,
            "visibility_partial_block_indices",
            "Optional increasing active-block indices that have partial visibility.",
            "I",
            OpSchema::Optional)
        .Input(
            13,
            "visibility_partial_block_mask",
            "Optional exact uint64 masks with shape (partial_blocks, query_block_size, ceil(key_block_size / 64)).",
            "tensor(uint64)",
            OpSchema::Optional)
        .Input(
            14,
            "attn_bias",
            "Optional additive packed bias with shape (active_blocks, query_block_size, key_block_size).",
            "T1",
            OpSchema::Optional)
        .Output(0, "Y", "Attention output, following ai.onnx.Attention layout rules.", "T1")
        .Output(
            1,
            "present_cache",
            "Updated logical cache snapshot.",
            "S",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            2,
            "qk_matmul_output",
            "Optional logical QK score tensor before masking and softmax.",
            "T1",
            OpSchema::Optional)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q, K, Y, and score types.")
        .TypeConstraint("T2", OpSchema::all_float_types_ir4(), "Constrain V types.")
        .TypeConstraint("U", OpSchema::all_non_complex_numeric_types_plus_bool_ir4(), "Constrain mask types.")
        .TypeConstraint("I", {"tensor(int32)", "tensor(int64)"}, "Constrain cumulative offsets and compact indices.")
        .TypeConstraint("P", {"tensor(int32)", "tensor(int64)"}, "Constrain absolute position types.")
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
        .Attr(
            "update_rule",
            "One of linear, gated, delta, or gated_delta.",
            AttributeProto::STRING,
            std::string("gated_delta"))
        .Attr("scale", "Output scale; zero derives 1/sqrt(d_k).", AttributeProto::FLOAT, 0.0f)
        .Attr("q_num_heads", "Number of query heads.", AttributeProto::INT)
        .Attr("kv_num_heads", "Number of compact key heads.", AttributeProto::INT)
        .Attr("v_num_heads", "Value/state head count; zero uses kv_num_heads.", AttributeProto::INT, int64_t{0})
        .Attr("chunk_size", "Implementation hint that does not change results.", AttributeProto::INT, int64_t{64})
        .Attr(
            "allow_lossy_state",
            "Whether implementation-selected lossy accumulator storage is permitted.",
            AttributeProto::INT,
            int64_t{0})
        .Input(0, "query", "Packed query tensor with shape (B, T, Hq*d_k).", "T")
        .Input(1, "key", "Packed key tensor with shape (B, T, Hkv*d_k).", "T")
        .Input(2, "value", "Packed value tensor with shape (B, T, Hv*d_v).", "T")
        .Input(
            3,
            "state",
            "Optional prior logical accumulator. Omission initializes zeros.",
            "S",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(4, "decay", "Optional decay gate required by gated update rules.", "T", OpSchema::Optional)
        .Input(5, "beta", "Optional update rate required by delta update rules.", "T", OpSchema::Optional)
        .Input(6, "sequence_lens", "Optional valid prefix lengths with shape (B).", "tensor(int64)", OpSchema::Optional)
        .Output(0, "output", "Packed recurrent attention output with shape (B, T, Ho*d_v).", "T")
        .Output(
            1,
            "present_state",
            "Updated logical recurrent accumulator snapshot.",
            "S",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
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
        .Attr(
            "allow_lossy_state",
            "Whether implementation-selected lossy history storage is permitted.",
            AttributeProto::INT,
            int64_t{0})
        .Input(0, "input", "Input tensor with shape (B, C, L).", "T")
        .Input(1, "weight", "Depthwise kernel with shape (C, 1, K).", "T")
        .Input(2, "bias", "Optional per-channel bias with shape (C).", "T", OpSchema::Optional)
        .Input(
            3,
            "state",
            "Optional prior logical history. Omission initializes zeros.",
            "S",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(4, "sequence_lens", "Optional valid prefix lengths with shape (B).", "tensor(int64)", OpSchema::Optional)
        .Output(0, "output", "Convolution output with shape (B, C, L).", "T")
        .Output(
            1,
            "present_state",
            "Updated bounded-history snapshot.",
            "S",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"}, "Constrain tensor types.")
        .TypeConstraint("S", {kCausalConvStateType}, "Nominal opaque causal-convolution state type.")
        .TypeAndShapeInferenceFunction(CausalConvWithStateInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic));

} // namespace ONNX_NAMESPACE
