// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/nn/utils.h"

#include <algorithm>

namespace ONNX_NAMESPACE {
namespace defs {
namespace nn {
namespace utils {

std::vector<int64_t> getConvPoolStrides(InferenceContext& ctx, size_t n_input_dims) {
  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
    if (std::any_of(strides.begin(), strides.end(), [](int64_t s) { return s <= 0; })) {
      fail_shape_inference("Attribute strides must only contain positive values");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }
  return strides;
}

void AttentionPropagateElemTypeFromInputToOutput(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Validate local_window_size early so invalid values are rejected during
  // model checking / shape inference, before the function body builder runs.
  const auto* const lws_attr = ctx.getAttribute("local_window_size");
  if (lws_attr != nullptr) {
    int64_t lws = lws_attr->i();
    if (lws != -1 && lws <= 0) {
      fail_shape_inference("local_window_size must be -1 or positive, got ", lws);
    }
  }

  int64_t kv_sequence_length = -1;
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  ONNX_NAMESPACE::TensorShapeProto qk_matmul_shape;
  if (hasInputShape(ctx, 0)) {
    const auto& query_shape = getInputShape(ctx, 0);
    const auto& query_dims = query_shape.dim();
    if ((query_dims.size() != 3) && (query_dims.size() != 4)) {
      fail_shape_inference("Inputs 0 (query) shall be 3 or 4 dimensions");
    }

    if (query_dims.size() == 3) {
      const auto* const q_num_heads_attr = ctx.getAttribute("q_num_heads");
      if (q_num_heads_attr == nullptr) {
        fail_type_inference("3D inputs expected to have q_num_heads attribute.");
      }
      const auto* const kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
      if (kv_num_heads_attr == nullptr) {
        fail_type_inference("3D inputs expected to have q_num_heads attribute.");
      }
    }

    *output_shape.add_dim() = query_dims[0]; // batch_size
    *output_shape.add_dim() = query_dims[1]; // num_heads for 4D, sequence_length for 3D

    *qk_matmul_shape.add_dim() = query_dims[0]; // batch_size

    if (hasInputShape(ctx, 1)) {
      const auto& key_shape = getInputShape(ctx, 1);
      const auto& key_dims = key_shape.dim();
      if ((key_dims.size() != 3) && (key_dims.size() != 4)) {
        fail_shape_inference("Inputs 1 (key) shall be 3 or 4 dimensions");
      }
    }

    if (hasInputShape(ctx, 2)) {
      const auto& value_shape = getInputShape(ctx, 2);
      const auto& value_dims = value_shape.dim();
      if ((value_dims.size() != 3) && (value_dims.size() != 4)) {
        fail_shape_inference("Inputs 2 (value) shall be 3 or 4 dimensions");
      }

      // Update Output Shape for 4D inputs
      // Input 0 (query) has shape (batch_size, q_num_heads, q_sequence_length, head_size)
      // Input 1 (key) has shape (batch_size, kv_num_heads, kv_sequence_length, head_size)
      // Input 2 (value) has shape (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
      // Output 0 has shape (batch_size, q_num_heads, q_sequence_length, v_head_size)
      if (value_dims.size() == 4 && query_dims.size() == 4) {
        kv_sequence_length = value_dims[2].dim_value();
        *output_shape.add_dim() = query_dims[2]; // sequence_length
        *output_shape.add_dim() = value_dims[3]; // head_size
        updateOutputShape(ctx, 0, output_shape);
        // Update qk_matmul_shape
        *qk_matmul_shape.add_dim() = query_dims[1]; // q_num_heads
        *qk_matmul_shape.add_dim() = query_dims[2]; // q_sequence_length
        qk_matmul_shape.add_dim()->set_dim_value(kv_sequence_length);
      }

      // Update Output Shape for 3D inputs
      // Input 0 (query) has shape (batch_size, q_sequence_length, q_hidden_size),
      // q_hidden_size = q_num_heads * head_size
      // Input 1 (key) has shape (batch_size, kv_sequence_length, k_hidden_size),
      // k_hidden_size = kv_num_heads * head_size
      // Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size),
      // v_hidden_size = kv_num_heads * v_head_size
      // Output 0 has shape (batch_size, q_sequence_length, hidden_size),
      // hidden_size = q_num_heads * v_head_size
      if (value_dims.size() == 3 && query_dims.size() == 3) {
        kv_sequence_length = value_dims[1].dim_value();
        const auto* const q_num_heads_attr = ctx.getAttribute("q_num_heads");
        if (q_num_heads_attr == nullptr) {
          fail_type_inference("3D inputs expected to have q_num_heads attribute.");
        }
        const auto* const kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
        if (kv_num_heads_attr == nullptr) {
          fail_type_inference("3D inputs expected to have kv_num_heads attribute.");
        }
        int64_t q_num_heads = q_num_heads_attr->i();
        int64_t kv_num_heads = kv_num_heads_attr->i();
        // Calculate v_head_size
        int64_t v_head_size = value_dims[2].dim_value() / kv_num_heads;
        output_shape.add_dim()->set_dim_value(v_head_size * q_num_heads);
        updateOutputShape(ctx, 0, output_shape);
        // Update qk_matmul_shape
        qk_matmul_shape.add_dim()->set_dim_value(q_num_heads);
        *qk_matmul_shape.add_dim() = query_dims[1];
        qk_matmul_shape.add_dim()->set_dim_value(kv_sequence_length);
      }
    }
  }

  if (ctx.hasOutput(3)) { // has qk_matmul_output
    propagateElemTypeFromInputToOutput(ctx, 0, 3);
    updateOutputShape(ctx, 3, qk_matmul_shape);
  }

  if (ctx.hasOutput(1) && ctx.hasOutput(2)) { // has present outputs
    if (ctx.hasInput(4) && ctx.hasInput(5)) { // has past_key
      // copy the type from query to present key and value
      propagateElemTypeFromInputToOutput(ctx, 4, 1);
      propagateElemTypeFromInputToOutput(ctx, 5, 2);

      if (hasInputShape(ctx, 4) && hasInputShape(ctx, 5)) {
        const auto& past_key_shape = getInputShape(ctx, 4);
        const auto& past_key_dims = past_key_shape.dim();
        const auto& past_value_shape = getInputShape(ctx, 5);
        const auto& past_value_dims = past_value_shape.dim();

        // past key has shape (batch_size, kv_num_heads, past_sequence_length, head_size)
        if (past_key_dims.size() != 4) {
          fail_shape_inference("The past_key input shall be 4 dimensions");
        }
        // past value has shape (batch_size, kv_num_heads, past_sequence_length, v_head_size)
        if (past_value_dims.size() != 4) {
          fail_shape_inference("The past_value input shall be 4 dimensions");
        }

        if (kv_sequence_length > 0 && past_key_dims[2].has_dim_value()) {
          int64_t total_sequence_length = kv_sequence_length + past_key_dims[2].dim_value();

          ONNX_NAMESPACE::TensorShapeProto present_key_shape;
          for (const auto& dim : past_key_dims) {
            *present_key_shape.add_dim() = dim;
          }

          ONNX_NAMESPACE::TensorShapeProto present_value_shape;
          for (const auto& dim : past_value_dims) {
            *present_value_shape.add_dim() = dim;
          }

          if (ctx.hasOutput(3)) { // has qk_matmul_output with bias
            qk_matmul_shape.mutable_dim(3)->set_dim_value(total_sequence_length);
            updateOutputShape(ctx, 3, qk_matmul_shape);
          }

          // shape of present key/value is (batch_size, kv_num_heads, total_sequence_length, head_size)
          present_key_shape.mutable_dim(2)->set_dim_value(total_sequence_length);
          present_value_shape.mutable_dim(2)->set_dim_value(total_sequence_length);

          updateOutputShape(ctx, 1, present_key_shape);
          updateOutputShape(ctx, 2, present_value_shape);
        }
      }
    }
  }
}

bool AttentionAppendFunctionCausalMask(const FunctionBodyBuildContext& ctx, FunctionBuilder& builder, bool padding) {
  builder.Add("NewKVSeqLen =  Shape <start = -2, end = -1> (PresentKey)")
      .Add("AttnBiasShape = Concat <axis = 0> (QSeqLen, NewKVSeqLen)");
  float neg_inf = -std::numeric_limits<float>::infinity();
  builder.Const1D("FloatNegInf", neg_inf);
  builder.Const1D("ScalarZero", 0.f);

  // If attn_mask is provided
  if (ctx.hasInput(3)) {
    const auto* const up = ctx.getInputType(3);
    if ((up == nullptr) || (!up->has_tensor_type()))
      return false;
    int64_t U = up->tensor_type().elem_type();
    builder.Add(
        U == ONNX_NAMESPACE::TensorProto_DataType_BOOL ? "AttnBiasShort = Where(attn_mask, ScalarZero, FloatNegInf)"
                                                       : "AttnBiasShort = Identity(attn_mask)");
    // If attn_mask has a shorter kv sequence length, we pad it to NewKVSeqLen with FloatNegInf
    if (padding) {
      builder.Add("MaskKVSeqLen = Shape <start = -1> (attn_mask)")
          .Add("PaddingKVSeqLen = Sub(NewKVSeqLen, MaskKVSeqLen)")
          .Add("Pads = Concat <axis = 0> (Zero1D, PaddingKVSeqLen)")
          .Add("FloatNegInfCast = CastLike(FloatNegInf, AttnBiasShort)")
          .Add("AttnBias = Pad(AttnBiasShort, Pads, FloatNegInfCast, NegOne1D)");
    } else {
      builder.Add("AttnBias = Identity(AttnBiasShort)");
    }
  } else {
    builder.Add("AttnBias = ConstantOfShape(AttnBiasShape)");
  }

  // If is_causal is set to true, causal masking is applied with bottom-right
  // (offset-aware) alignment: a query at in-block index i attends key j iff
  // j <= i + offset, where offset is the number of valid keys preceding the query block.
  // For an internal past_key cache offset is the scalar PastKVSeqLen; for an external
  // (static) cache (nonpad_kv_seqlen present, no past_key) offset is per batch and the
  // builder scope holds CausalOffsetPerBatch (= nonpad_kv_seqlen - q_len).
  // When both attn_mask and is_causal are set, the two are combined: a boolean
  // attn_mask intersects with the causal frontier (a position is attended only if
  // allowed by both), while a float attn_mask is added as a bias to the causal
  // bias rather than strictly disabling positions (matching defs.cc).
  const auto* const is_causal_attr = ctx.getAttribute("is_causal");
  int64_t is_causal = (is_causal_attr != nullptr) ? is_causal_attr->i() : 0;
  const bool external_cache_offset = (is_causal == 1) && ctx.hasInput(6) && !ctx.hasInput(4);
  if (is_causal == 1) {
    builder.Const1D("Zero", static_cast<int64_t>(0))
        .Const1D("One", static_cast<int64_t>(1))
        .Add("ZeroNoDim = Squeeze(Zero, Zero)")
        .Add("OneNoDim = Squeeze(One, Zero)")
        .Add("SequenceLength = Gather(AttnBiasShape, ZeroNoDim)")
        .Add("TotalSequenceLength = Gather(AttnBiasShape, OneNoDim)")
        .Add("RangeRow = Range(ZeroNoDim, SequenceLength, OneNoDim)")
        .Add("RangeRow2D = Unsqueeze(RangeRow, One)")
        .Add("RangeCol = Range(ZeroNoDim, TotalSequenceLength, OneNoDim)")
        .Add("RangeCol2D = Unsqueeze(RangeCol, Zero)");
    if (external_cache_offset) {
      // Per-batch bottom-right frontier: broadcast the (batch,) offset into a 4D
      // (batch, 1, q, total) boolean mask so it composes with the (batch,1,1,kv)
      // padding mask added by the caller.
      builder.Const("Axes01", std::vector<int64_t>{0, 1})
          .Const("Axes123", std::vector<int64_t>{1, 2, 3})
          .Add("RangeRow4D = Unsqueeze(RangeRow2D, Axes01)") // (1, 1, q, 1)
          .Add("RangeCol4D = Unsqueeze(RangeCol2D, Axes01)") // (1, 1, 1, total)
          .Add("OffsetB4D = Unsqueeze(CausalOffsetPerBatch, Axes123)") // (batch, 1, 1, 1)
          .Add("RowPlusOff = Add(RangeRow4D, OffsetB4D)") // (batch, 1, q, 1)
          .Add("BoolMaskTri = Less(RowPlusOff, RangeCol4D)"); // (batch, 1, q, total)
      // MaskTri is 4D (batch, 1, q, total).  AttnBias may be 3D (batch, q, kv)
      // which does not broadcast correctly against a 4D tensor (ONNX left-pads
      // with 1s, giving (1, batch, q, kv) instead of (batch, 1, q, kv)).
      // Promote AttnBias to 4D at build time based on the known rank of attn_mask.
      int causal_mask_rank = -1;
      if (ctx.hasInput(3)) {
        const auto* mask_tp = ctx.getInputType(3);
        if (mask_tp && mask_tp->has_tensor_type() && mask_tp->tensor_type().has_shape()) {
          causal_mask_rank = mask_tp->tensor_type().shape().dim_size();
        }
      }
      if (causal_mask_rank == 3) {
        builder.Add("AttnBias4DCausal = Unsqueeze(AttnBias, One)");
      } else if (causal_mask_rank == 4) {
        builder.Add("AttnBias4DCausal = Identity(AttnBias)");
      } else {
        builder.Add("CausalBiasShape = Concat <axis = 0> (NegOne1D, One1D, QSeqLen, NewKVSeqLen)")
            .Add("AttnBias4DCausal = Reshape(AttnBias, CausalBiasShape)");
      }
      builder.Add("MaskTri = Where(BoolMaskTri, FloatNegInf, ScalarZero)")
          .Add("AttnBiasCausalOrNot = Add(AttnBias4DCausal, MaskTri)");
    } else {
      // Internal cache / no cache: scalar offset (PastKVSeqLen), 2D (q, total) mask.
      builder.Add("RangeRow2DPast = Add(RangeRow2D, PastKVSeqLen)")
          .Add("BoolMaskTri = Less(RangeRow2DPast, RangeCol2D)");
      builder.Add("MaskTri = Where(BoolMaskTri, FloatNegInf, ScalarZero)")
          .Add("AttnBiasCausalOrNot = Add(AttnBias, MaskTri)");
    }
  } else {
    builder.Add("AttnBiasCausalOrNot = Identity(AttnBias)");
  }
  return true;
}

} // namespace utils
} // namespace nn
} // namespace defs
} // namespace ONNX_NAMESPACE
