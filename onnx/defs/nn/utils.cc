/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include "onnx/defs/nn/utils.h"

#include <algorithm>

namespace ONNX_NAMESPACE {
namespace defs {
namespace nn {
namespace utils {

void AttentionPropagateElemTypeFromInputToOutput(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  int64_t kv_sequence_length = -1;
  ONNX_NAMESPACE::TensorShapeProto output_shape;
  ONNX_NAMESPACE::TensorShapeProto qk_matmul_shape;
  if (hasInputShape(ctx, 0)) {
    auto& query_shape = getInputShape(ctx, 0);
    auto& query_dims = query_shape.dim();
    if ((query_dims.size() != 3) && (query_dims.size() != 4)) {
      fail_shape_inference("Inputs 0 (query) shall be 3 or 4 dimensions");
    }

    if (query_dims.size() == 3) {
      auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
      if (q_num_heads_attr == nullptr) {
        fail_type_inference("3D inputs expected to have q_num_heads attribute.");
      }
      auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
      if (kv_num_heads_attr == nullptr) {
        fail_type_inference("3D inputs expected to have q_num_heads attribute.");
      }
    }

    *output_shape.add_dim() = query_dims[0]; // batch_size
    *output_shape.add_dim() = query_dims[1]; // num_heads for 4D, sequence_length for 3D

    *qk_matmul_shape.add_dim() = query_dims[0]; // batch_size

    if (hasInputShape(ctx, 1)) {
      auto& key_shape = getInputShape(ctx, 1);
      auto& key_dims = key_shape.dim();
      if ((key_dims.size() != 3) && (key_dims.size() != 4)) {
        fail_shape_inference("Inputs 1 (key) shall be 3 or 4 dimensions");
      }
    }

    if (hasInputShape(ctx, 2)) {
      auto& value_shape = getInputShape(ctx, 2);
      auto& value_dims = value_shape.dim();
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
        auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
        if (q_num_heads_attr == nullptr) {
          fail_type_inference("3D inputs expected to have q_num_heads attribute.");
        }
        auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
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
        auto& past_key_shape = getInputShape(ctx, 4);
        auto& past_key_dims = past_key_shape.dim();
        auto& past_value_shape = getInputShape(ctx, 5);
        auto& past_value_dims = past_value_shape.dim();

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
          for (auto& dim : past_key_dims) {
            *present_key_shape.add_dim() = dim;
          }

          ONNX_NAMESPACE::TensorShapeProto present_value_shape;
          for (auto& dim : past_value_dims) {
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
    auto* up = ctx.getInputType(3);
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

  // If is_causal set to true, the attention masking is a lower triangular matrix when the mask
  // is a square matrix. The attention masking has the form of the upper left causal bias due to
  // the alignment when the mask is a non-square matrix.
  // An error is thrown if both attn_mask and is_causal are set.
  auto* is_causal_attr = ctx.getAttribute("is_causal");
  int64_t is_causal = (is_causal_attr != nullptr) ? is_causal_attr->i() : 0;
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
        .Add("RangeCol2D = Unsqueeze(RangeCol, Zero)")
        .Add("RangeRow2DPast = Add(RangeRow2D, PastKVSeqLen)")
        .Add("BoolMaskTri = Less(RangeRow2DPast, RangeCol2D)")
        .Add("MaskTri = Where(BoolMaskTri, FloatNegInf, ScalarZero)")
        .Add("AttnBiasCausalOrNot = Add(AttnBias, MaskTri)");
  } else {
    builder.Add("AttnBiasCausalOrNot = Identity(AttnBias)");
  }
  return true;
}

} // namespace utils
} // namespace nn
} // namespace defs
} // namespace ONNX_NAMESPACE
