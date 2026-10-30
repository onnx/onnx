// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/nn/utils.h"

#include <algorithm>
#include <string>

#include "onnx/defs/doc_strings.h"

namespace ONNX_NAMESPACE::defs::nn::utils {

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
        fail_type_inference("3D inputs expected to have kv_num_heads attribute.");
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
        if (q_num_heads <= 0 || kv_num_heads <= 0) {
          fail_type_inference("q_num_heads and kv_num_heads must be positive.");
        }
        // Calculate v_head_size
        int64_t v_head_size = checkedDivide(value_dims[2].dim_value(), kv_num_heads);
        output_shape.add_dim()->set_dim_value(checkedMultiply(v_head_size, q_num_heads));
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

  if ((ctx.hasOutput(1) && ctx.hasOutput(2)) && (ctx.hasInput(4) && ctx.hasInput(5))) // has present outputs
  { // has past_key
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
        int64_t total_sequence_length = checkedAdd(kv_sequence_length, past_key_dims[2].dim_value());

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

bool AttentionAppendFunctionCausalMask(
    const FunctionBodyBuildContext& ctx,
    FunctionBuilder& builder,
    bool padding,
    bool cast_mask_to_bias) {
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
    } else {
      // Internal cache / no cache: scalar offset (PastKVSeqLen), 2D (q, total) mask.
      builder.Add("RangeRow2DPast = Add(RangeRow2D, PastKVSeqLen)")
          .Add("BoolMaskTri = Less(RangeRow2DPast, RangeCol2D)");
    }
    if (cast_mask_to_bias) {
      // Cast the generated float mask before Add so non-float32 bias inputs
      // remain valid.
      builder.Add("MaskTriFloat = Where(BoolMaskTri, FloatNegInf, ScalarZero)")
          .Add("MaskTri = CastLike(MaskTriFloat, AttnBias)");
    } else {
      // Preserve the exact function body emitted for frozen Attention-24.
      builder.Add("MaskTri = Where(BoolMaskTri, FloatNegInf, ScalarZero)");
    }
    builder.Add("AttnBiasCausalOrNot = Add(AttnBias, MaskTri)");
  } else {
    builder.Add("AttnBiasCausalOrNot = Identity(AttnBias)");
  }
  return true;
}

static bool BuildLinearAttentionFunction(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // LinearAttention <update_rule, scale, q_num_heads, kv_num_heads, chunk_size>
  //   (query, key, value, past_state?, decay?, beta?)
  //   => (output, present_state)
  //
  // Reference decomposition (sequential recurrence via Scan):
  //   1. Reshape/transpose 3D packed inputs (B, T, H*D) -> 4D (B, H, T, D)
  //   2. Initialize state (past_state or zeros), shape (B, H_kv, d_k, d_v)
  //   3. Compute scale factor (attribute or 1/sqrt(d_k))
  //   4. Run Scan over T axis with body implementing one recurrence step
  //   5. Transpose/reshape 4D outputs back to 3D (B, T, H_q*d_v)

  // --- Step 1: Extract attributes and validate ---
  const auto* update_rule_attr = ctx.getAttribute("update_rule");
  std::string update_rule =
      (update_rule_attr != nullptr && update_rule_attr->has_s()) ? update_rule_attr->s() : "gated_delta";

  const auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
  const auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
  if (q_num_heads_attr == nullptr || !q_num_heads_attr->has_i() || kv_num_heads_attr == nullptr ||
      !kv_num_heads_attr->has_i()) {
    return false;
  }
  int64_t q_num_heads = q_num_heads_attr->i();
  int64_t kv_num_heads = kv_num_heads_attr->i();
  const int64_t input_q_heads = q_num_heads;
  const int64_t input_k_heads = kv_num_heads;
  const auto* v_heads_attr = ctx.getAttribute("v_num_heads");
  const int64_t v_heads = v_heads_attr == nullptr ? 0 : v_heads_attr->i();
  if (v_heads < 0 ||
      (v_heads > 0 && (q_num_heads != kv_num_heads || kv_num_heads <= 0 || v_heads % kv_num_heads != 0))) {
    return false;
  }
  if (v_heads > 0) {
    q_num_heads = v_heads;
    kv_num_heads = v_heads;
  }

  const auto* scale_attr = ctx.getAttribute("scale");
  float scale = (scale_attr != nullptr && scale_attr->has_f()) ? scale_attr->f() : 0.0f;

  // Per-rule presence of optional inputs
  bool has_decay = (update_rule == "gated" || update_rule == "gated_delta");
  bool has_beta = (update_rule == "delta" || update_rule == "gated_delta");
  if ((update_rule != "linear" && update_rule != "gated" && update_rule != "delta" && update_rule != "gated_delta") ||
      ctx.hasInput(4) != has_decay || ctx.hasInput(5) != has_beta) {
    return false;
  }

  // Need query type (for CastLike on zero state) — checked indirectly via input 0 presence.
  if (ctx.getInputType(0) == nullptr) {
    return false;
  }

  // GQA: q_num_heads must be a positive multiple of kv_num_heads.
  // group_size == 1 is the standard MHA case (no expansion); group_size > 1
  // means each KV head is shared by `group_size` query heads.
  if (q_num_heads <= 0 || kv_num_heads <= 0 || q_num_heads % kv_num_heads != 0) {
    return false;
  }
  int64_t group_size = q_num_heads / kv_num_heads;

  FunctionBuilder builder(functionProto);

  // --- Step 2: Pre-Scan 3D -> 4D reshape and transpose ---
  // Inputs arrive packed as (B, T, H*D). Reshape to (B, T, H, D) then
  // transpose to (B, H, T, D) so the Scan iterates along the T axis.
  builder.Const1D("NegOne1D", static_cast<int64_t>(-1))
      .Const1D("One1D", static_cast<int64_t>(1))
      .Const1D("HqConst", q_num_heads)
      .Const1D("HkvConst", kv_num_heads)
      .Const1D("InputHq", input_q_heads)
      .Const1D("InputHk", input_k_heads)
      .Add(R"ONNX(
                BatchDim = Shape <start = 0, end = 1> (query)
                SeqLenDim = Shape <start = 1, end = 2> (query)
                QLastDim = Shape <start = 2, end = 3> (query)
                VLastDim = Shape <start = 2, end = 3> (value)
                DkDim = Div (QLastDim, InputHq)
                DvDim = Div (VLastDim, HkvConst)
              )ONNX");

  // Inputs are reshaped to (B, T, H, D), then transposed so the T axis is
  // first: (T, B, H, D). The leading T axis is consumed by Scan (which
  // only supports scan_input_axes=0 in the ONNX reference runtime).

  // Query: (B, T, H_q*d_k) -> (B, T, H_q, d_k) -> (T, B, H_q, d_k)
  builder.Add(R"ONNX(
                QShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, InputHq, DkDim)
                QReshaped = Reshape <allowzero = 1> (query, QShape4D)
                QCompact = Transpose <perm = [1, 0, 2, 3]> (QReshaped)
              )ONNX");

  // Keep compact key heads separate from value/state heads.
  builder.Add(R"ONNX(
                KShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, InputHk, DkDim)
                VShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, HkvConst, DvDim)
                KReshaped = Reshape <allowzero = 1> (key, KShape4D)
                KCompact = Transpose <perm = [1, 0, 2, 3]> (KReshaped)
                VReshaped = Reshape <allowzero = 1> (value, VShape4D)
                V4D = Transpose <perm = [1, 0, 2, 3]> (VReshaped)
              )ONNX");

  if (v_heads > 0) {
    std::vector<int64_t> head_indices;
    head_indices.reserve(static_cast<std::size_t>(v_heads));
    for (int64_t h = 0; h < v_heads; ++h) {
      head_indices.push_back(h / (v_heads / input_k_heads));
    }
    builder.Const("HeadIndices", head_indices).Add(R"ONNX(
                  Q4D = Gather <axis = 2> (QCompact, HeadIndices)
                  K4D = Gather <axis = 2> (KCompact, HeadIndices)
                )ONNX");
  } else {
    builder.Add("Q4D = Identity (QCompact)\nK4D = Identity (KCompact)");
  }

  if (has_decay) {
    // Decay: (B, T, H_kv*d_k) -> (T, B, H_kv, d_k), or
    //        (B, T, H_kv)      -> (T, B, H_kv, 1)   [broadcastable].
    builder.Add(R"ONNX(
                  DecayLastDim = Shape <start = 2, end = 3> (decay)
                  DecayWidth = Div (DecayLastDim, HkvConst)
                  DecayShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, HkvConst, DecayWidth)
                  DecayReshaped = Reshape <allowzero = 1> (decay, DecayShape4D)
                  Decay4D = Transpose <perm = [1, 0, 2, 3]> (DecayReshaped)
                )ONNX");
  }

  if (has_beta) {
    // Beta: (B, T, H_kv) or (B, T, 1) -> reshape (B, T, -1, 1)
    //       -> transpose (T, B, H_kv_or_1, 1) [broadcastable head dim].
    builder.Add(R"ONNX(
                  BetaHeads = Shape <start = 2, end = 3> (beta)
                  BetaShape4D = Concat <axis = 0> (BatchDim, SeqLenDim, BetaHeads, One1D)
                  BetaReshaped = Reshape <allowzero = 1> (beta, BetaShape4D)
                  Beta4D = Transpose <perm = [1, 0, 2, 3]> (BetaReshaped)
                )ONNX");
  }

  // --- Step 3: Initialize recurrence state (B, H_kv, d_k, d_v) ---
  // Compute d_k and d_v as 1D shape tensors for downstream use
  // (state shape here, scale factor in Step 4).
  // State and scale are kept in float32 inside the body to match the
  // Python reference's fp32 accumulation. `present_state` is cast back
  // to S (past_state's dtype, or query's dtype when past_state is
  // absent) after the Scan; `output` is cast back to T inside the body.
  if (ctx.hasInput(3)) {
    // past_state has shape (B, H_kv, d_k, d_v) and dtype S.
    // Upcast to float32 for accumulation regardless of S.
    builder.Add("State0 = Cast <to = 1> (past_state)");
  } else {
    // ConstantOfShape with default value yields float32 already; no
    // cast required. S anchor for the post-Scan cast defaults to T
    // (query's dtype) in this case.
    builder.Add(R"ONNX(
                  StateShape = Concat <axis = 0> (BatchDim, HkvConst, DkDim, DvDim)
                  State0 = ConstantOfShape (StateShape)
                )ONNX");
  }

  // --- Step 4: Compute scale factor (scalar, float32) ---
  // If `scale` attribute is non-zero, use it directly; otherwise
  // derive 1/sqrt(d_k) from query's last dim. ScaleFactor stays in
  // float32 to multiply against the float32 read result inside the
  // body; the body casts the final output back to T.
  if (scale != 0.0f) {
    builder.Const("ScaleFactor", scale);
  } else {
    builder.Add(R"ONNX(
                  DkScalar = Squeeze (DkDim)
                  DkFloat = Cast <to = 1> (DkScalar)
                  SqrtDk = Sqrt (DkFloat)
                  ScaleFactor = Reciprocal (SqrtDk)
                )ONNX");
  }

  // --- Step 4b: Precompute GQA expand/read shapes (only when group_size > 1) ---
  // GQA expansion uses the same Unsqueeze + Expand + Reshape pattern as the
  // standard Attention op. The state lives at H_kv heads (one state per
  // KV head — recurrence is per-KV-head); only the read step needs to
  // broadcast it across the `group_size` query heads sharing that KV head.
  // Shapes are computed once outside the Scan body and captured inside
  // by name.
  if (group_size > 1) {
    builder.Const1D("GroupSize", group_size).Const1D("Two1D", static_cast<int64_t>(2)).Add(R"ONNX(
                  StateExpandShape = Concat <axis = 0> (BatchDim, HkvConst, GroupSize, DkDim, DvDim)
                  StateReadShape = Concat <axis = 0> (BatchDim, HqConst, DkDim, DvDim)
                )ONNX");
  }

  // --- Step 5: Scan node with body subgraph ---
  // The body implements one recurrence step. We write one complete
  // body per update_rule (linear / gated / delta / gated_delta) so
  // the recurrence is readable inline rather than assembled from
  // fragments. ScaleFactor (and, for group_size > 1,
  // StateExpandShape / StateReadShape) are referenced from the
  // enclosing scope (Scan subgraphs may capture outer-scope
  // tensors by name).
  //
  // The body always accumulates in float32 to match the Python
  // reference: each sliced T-typed input (q_t/k_t/v_t/decay_t/beta_t)
  // is cast to float32 at body entry; state_in/state_out stay
  // in float32 throughout; output_t is cast back to T (anchored on
  // q_t) before being emitted as a per-step Scan output. The post-
  // Scan code casts FinalState back to S.
  //
  // Axes constants used inside the body for Unsqueeze/Squeeze:
  // NegOne1D was added in Step 2; Two1D is set up in Step 4b
  // when group_size > 1.
  builder.Const1D("NegTwo1D", static_cast<int64_t>(-2));

  // Read block (shared across all update_rules; differs only by GQA):
  // output_t = ScaleFactor * (q_t^T @ state_out), then CastLike to T.
  // For GQA (group_size > 1) the per-KV-head state must first be
  // repeat-interleaved across the `group_size` query heads sharing
  // each KV head, using the same Unsqueeze + Expand + Reshape
  // pattern as the standard Attention op. `StateExpandShape` and
  // `StateReadShape` are captured from the enclosing scope.
  const std::string read_block = (group_size > 1) ? R"ONNX(
                StateUnsq = Unsqueeze (state_out, Two1D)
                StateExpanded = Expand (StateUnsq, StateExpandShape)
                StateForRead = Reshape <allowzero = 1> (StateExpanded, StateReadShape)
                QtRow = Unsqueeze (q_t_f, NegTwo1D)
                ReadOut = MatMul (QtRow, StateForRead)
                ReadSq = Squeeze (ReadOut, NegTwo1D)
                output_t_f = Mul (ScaleFactor, ReadSq)
                output_t = CastLike (output_t_f, q_t)
              )ONNX"
                                                  : R"ONNX(
                QtRow = Unsqueeze (q_t_f, NegTwo1D)
                ReadOut = MatMul (QtRow, state_out)
                ReadSq = Squeeze (ReadOut, NegTwo1D)
                output_t_f = Mul (ScaleFactor, ReadSq)
                output_t = CastLike (output_t_f, q_t)
              )ONNX";

  // One complete recurrence body per update_rule. Each body:
  //   * casts its T-typed inputs to float32,
  //   * computes state_out = (decayed) state_in + KtCol @ WriteRow,
  //     where WriteRow comes from v_t_f directly (linear/gated) or
  //     from a delta-rule correction Delta = beta * (v - retrieved).
  std::string body;
  std::string body_inputs;
  std::string scan_inputs;
  int64_t num_scan_inputs = 0;
  if (update_rule == "linear") {
    body_inputs = "state_in, q_t, k_t, v_t";
    scan_inputs = "State0, Q4D, K4D, V4D";
    num_scan_inputs = 3;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                WriteRow = Unsqueeze (v_t_f, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (state_in, Outer)
              )ONNX";
  } else if (update_rule == "gated") {
    body_inputs = "state_in, q_t, k_t, v_t, decay_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Decay4D";
    num_scan_inputs = 4;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                decay_t_f = Cast <to = 1> (decay_t)
                DecayExp = Exp (decay_t_f)
                DecayExpUnsq = Unsqueeze (DecayExp, NegOne1D)
                DecayedState = Mul (state_in, DecayExpUnsq)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                WriteRow = Unsqueeze (v_t_f, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (DecayedState, Outer)
              )ONNX";
  } else if (update_rule == "delta") {
    body_inputs = "state_in, q_t, k_t, v_t, beta_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Beta4D";
    num_scan_inputs = 4;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                beta_t_f = Cast <to = 1> (beta_t)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                StateT = Transpose <perm = [0, 1, 3, 2]> (state_in)
                Retrieved = MatMul (StateT, KtCol)
                RetrievedSq = Squeeze (Retrieved, NegOne1D)
                Diff = Sub (v_t_f, RetrievedSq)
                Delta = Mul (beta_t_f, Diff)
                WriteRow = Unsqueeze (Delta, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (state_in, Outer)
              )ONNX";
  } else { // gated_delta
    body_inputs = "state_in, q_t, k_t, v_t, decay_t, beta_t";
    scan_inputs = "State0, Q4D, K4D, V4D, Decay4D, Beta4D";
    num_scan_inputs = 5;
    body = R"ONNX(
                q_t_f = Cast <to = 1> (q_t)
                k_t_f = Cast <to = 1> (k_t)
                v_t_f = Cast <to = 1> (v_t)
                decay_t_f = Cast <to = 1> (decay_t)
                beta_t_f = Cast <to = 1> (beta_t)
                DecayExp = Exp (decay_t_f)
                DecayExpUnsq = Unsqueeze (DecayExp, NegOne1D)
                DecayedState = Mul (state_in, DecayExpUnsq)
                KtCol = Unsqueeze (k_t_f, NegOne1D)
                StateT = Transpose <perm = [0, 1, 3, 2]> (DecayedState)
                Retrieved = MatMul (StateT, KtCol)
                RetrievedSq = Squeeze (Retrieved, NegOne1D)
                Diff = Sub (v_t_f, RetrievedSq)
                Delta = Mul (beta_t_f, Diff)
                WriteRow = Unsqueeze (Delta, NegTwo1D)
                Outer = MatMul (KtCol, WriteRow)
                state_out = Add (DecayedState, Outer)
              )ONNX";
  }

  // Assemble Scan node text. Body name is arbitrary.
  // We omit scan_input_axes/scan_output_axes (default 0); Q/K/V/decay/beta
  // were already pre-transposed so the T axis is the leading dimension.
  std::string scan_text =
      "ScanFinalState, ScanOutput = Scan <\n"
      "  num_scan_inputs = " +
      std::to_string(num_scan_inputs) +
      ",\n"
      "  body = linear_attn_step (" +
      body_inputs + ") => (state_out, output_t) {\n" + body + read_block +
      "              }\n"
      "> (" +
      scan_inputs + ")";

  builder.Const("ZeroLength", static_cast<int64_t>(0));
  builder.Add(R"ONNX(
                SeqLenScalar = Squeeze (SeqLenDim)
                IsEmpty = Equal (SeqLenScalar, ZeroLength)
                EmptyOutputShape = Concat <axis = 0> (SeqLenDim, BatchDim, HqConst, DvDim)
              )ONNX");
  const std::string guarded_scan = R"ONNX(
                FinalState, OutputAccum = If (IsEmpty) <
                  then_branch = empty_sequence () => (EmptyState, EmptyOutput) {
                    EmptyState = Identity (State0)
                    EmptyFloat = ConstantOfShape (EmptyOutputShape)
                    EmptyOutput = CastLike (EmptyFloat, query)
                  },
                  else_branch = nonempty_sequence () => (ScanFinalState, ScanOutput) {
              )ONNX" +
      scan_text + "\n}>";
  builder.Add(guarded_scan.c_str());

  // --- Step 6: Post-Scan 4D -> 3D reshape and present_state ---
  // OutputAccum: (T, B, H_q, d_v) -> Transpose to (B, T, H_q, d_v)
  // -> Reshape to (B, T, H_q * d_v). Already T-typed (cast inside body).
  // present_state: cast FinalState (float32) back to S.
  //   * With past_state: anchor on past_state's dtype.
  //   * Without past_state: default S to T (query's dtype).
  builder.Add(R"ONNX(
                OutputTransposed = Transpose <perm = [1, 0, 2, 3]> (OutputAccum)
                OutputWidth = Mul (HqConst, DvDim)
                OutputShape3D = Concat <axis = 0> (BatchDim, SeqLenDim, OutputWidth)
                output = Reshape <allowzero = 1> (OutputTransposed, OutputShape3D)
              )ONNX");
  if (ctx.hasInput(3)) {
    builder.Add("present_state = CastLike (FinalState, past_state)");
  } else {
    builder.Add("present_state = CastLike (FinalState, query)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

static void InferLinearAttentionShape(InferenceContext& ctx, bool expanded_value_heads) {
  // Read required attributes
  const auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
  const auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
  int64_t q_num_heads = (q_num_heads_attr && q_num_heads_attr->has_i()) ? q_num_heads_attr->i() : 0;
  int64_t kv_num_heads = (kv_num_heads_attr && kv_num_heads_attr->has_i()) ? kv_num_heads_attr->i() : 0;

  // GQA constraint: q_num_heads must be a positive multiple of kv_num_heads.
  // q_num_heads == kv_num_heads is MHA; q_num_heads > kv_num_heads with the
  // divisibility rule is GQA (or MQA when kv_num_heads == 1). Non-positive
  // head counts are rejected here so that downstream divisions and the
  // function body always see strictly positive values.
  if (q_num_heads <= 0) {
    fail_type_inference("q_num_heads must be > 0, got ", q_num_heads);
  }
  if (kv_num_heads <= 0) {
    fail_type_inference("kv_num_heads must be > 0, got ", kv_num_heads);
  }
  if (q_num_heads % kv_num_heads != 0) {
    fail_type_inference("q_num_heads (", q_num_heads, ") must be a multiple of kv_num_heads (", kv_num_heads, ")");
  }
  const int64_t v_heads = getAttribute(ctx, "v_num_heads", static_cast<int64_t>(0));
  if (v_heads < 0 || (v_heads > 0 && (q_num_heads != kv_num_heads || v_heads % kv_num_heads != 0))) {
    fail_shape_inference(
        "v_num_heads must be zero, or a positive multiple of kv_num_heads with q_num_heads == kv_num_heads");
  }
  const int64_t state_heads = v_heads == 0 ? kv_num_heads : v_heads;
  const int64_t output_heads = v_heads == 0 ? q_num_heads : v_heads;

  // Read update_rule attribute with default
  const auto* update_rule_attr = ctx.getAttribute("update_rule");
  std::string update_rule = (update_rule_attr != nullptr) ? update_rule_attr->s() : "gated_delta";

  // Validate update_rule value
  if (update_rule != "linear" && update_rule != "gated" && update_rule != "delta" && update_rule != "gated_delta") {
    fail_type_inference("update_rule must be one of: 'linear', 'gated', 'delta', 'gated_delta'");
  }

  // Validate update_rule vs optional inputs (decay=input 4, beta=input 5).
  const bool has_past_state = ctx.hasInput(3);
  const bool has_decay = ctx.hasInput(4);
  const bool has_beta = ctx.hasInput(5);

  if (update_rule == "linear") {
    if (has_decay) {
      fail_type_inference("update_rule 'linear' forbids decay input");
    }
    if (has_beta) {
      fail_type_inference("update_rule 'linear' forbids beta input");
    }
  } else if (update_rule == "gated") {
    if (!has_decay) {
      fail_type_inference("update_rule 'gated' requires decay input");
    }
    if (has_beta) {
      fail_type_inference("update_rule 'gated' forbids beta input");
    }
  } else if (update_rule == "delta") {
    if (has_decay) {
      fail_type_inference("update_rule 'delta' forbids decay input");
    }
    if (!has_beta) {
      fail_type_inference("update_rule 'delta' requires beta input");
    }
  } else if (update_rule == "gated_delta") {
    if (!has_decay) {
      fail_type_inference("update_rule 'gated_delta' requires decay input");
    }
    if (!has_beta) {
      fail_type_inference("update_rule 'gated_delta' requires beta input");
    }
  }

  // --- Type propagation ---
  // output: type T from query.
  //
  // present_state uses the separately-declared state type S only when S can be
  // anchored by the optional past_state input. When past_state is omitted,
  // type inference cannot independently infer S, so present_state explicitly
  // falls back to type T from query on the first step. Callers that require an
  // fp32 state while using lower-precision query activations must therefore
  // provide past_state with the desired state element type.
  // (The spec recommends S == float32 or S == T for numerical stability.)
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (has_past_state && ctx.getInputType(3) != nullptr && ctx.getInputType(3)->has_tensor_type()) {
    propagateElemTypeFromInputToOutput(ctx, 3, 1);
  } else {
    // No past_state is available to anchor S, so present_state defaults to T.
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
  }

  // --- Shape inference via dimension unification ---
  // Declare logical dims and unify each input against its expected shape.
  // unifyInputShape validates the rank of each provided input and merges
  // the most-informative known value across inputs that share a dim
  // (B and T across Q/K/V/decay/beta/past_state; state heads between past_state
  // and the head-count attributes; d_k between past_state, Q (via
  // QPack/q_num_heads) and K; d_v between past_state and V).
  Dim B, T, QPack, KPack, VPack, Hstate, Dk, Dv;
  Hstate.set_dim_value(state_heads);
  ctx.unifyInputShape(0, {B, T, QPack});
  ctx.unifyInputShape(1, {B, T, KPack});
  ctx.unifyInputShape(2, {B, T, VPack});
  if (has_past_state) {
    ctx.unifyInputShape(3, {B, Hstate, Dk, Dv});
  }
  // decay shape: (B, T, Hstate * d_k) for per-key-dim or (B, T, Hstate) for
  // per-head. Both are rank-3; we only constrain B and T.
  if (has_decay) {
    Dim DecayLast;
    ctx.unifyInputShape(4, {B, T, DecayLast});
  }
  // beta shape: (B, T, Hstate) or (B, T, 1). Both rank-3; only B, T checked.
  if (has_beta) {
    Dim BetaLast;
    ctx.unifyInputShape(5, {B, T, BetaLast});
  }

  // Derive d_k from Q (via q_num_heads) and K (via kv_num_heads).
  // Multiple sources for the same logical dim are unified, so an
  // inconsistency between Q and K (e.g. mismatched d_k) fails here.
  // Reject packed last dims that are not divisible by their head count
  // up-front so the resulting integer-division d_k actually corresponds
  // to a shape the function body can reshape to (B, T, H, D).
  if (QPack.has_dim_value()) {
    if (QPack.dim_value() % q_num_heads != 0) {
      fail_shape_inference(
          "Packed query last dim (", QPack.dim_value(), ") must be divisible by q_num_heads (", q_num_heads, ")");
    }
    unifyDim(Dk, checkedDivide(QPack.dim_value(), q_num_heads));
  }
  if (KPack.has_dim_value()) {
    if (KPack.dim_value() % kv_num_heads != 0) {
      fail_shape_inference(
          "Packed key last dim (", KPack.dim_value(), ") must be divisible by kv_num_heads (", kv_num_heads, ")");
    }
    unifyDim(Dk, checkedDivide(KPack.dim_value(), kv_num_heads));
  }
  // Derive d_v from V.
  if (VPack.has_dim_value()) {
    if (VPack.dim_value() % state_heads != 0) {
      fail_shape_inference(
          "Packed value last dim (",
          VPack.dim_value(),
          ") must be divisible by the value head count (",
          state_heads,
          ")");
    }
    unifyDim(Dv, checkedDivide(VPack.dim_value(), state_heads));
  }

  // Strict validation of decay/beta last dims, deferred until d_k and Hstate
  // are derived above. The earlier unifyInputShape calls only checked rank
  // and B/T; here we reject concrete last dims that the function body and
  // Python reference cannot broadcast correctly.
  //   decay last dim must be Hstate (per-head scalar form) or Hstate * d_k
  //     (per-key-dim form). When d_k is unknown we still require divisibility
  //     by Hstate.
  //   beta  last dim must be 1 (broadcast) or Hstate (per-head).
  if (expanded_value_heads &&
      ((Dk.has_dim_value() && Dk.dim_value() <= 0) || (Dv.has_dim_value() && Dv.dim_value() <= 0))) {
    fail_shape_inference("Key and value head dimensions must be positive");
  }
  if (has_decay) {
    const auto* decay_type = ctx.getInputType(4);
    if (decay_type != nullptr && decay_type->has_tensor_type() && decay_type->tensor_type().has_shape() &&
        decay_type->tensor_type().shape().dim_size() == 3 && decay_type->tensor_type().shape().dim(2).has_dim_value()) {
      const int64_t decay_last = decay_type->tensor_type().shape().dim(2).dim_value();
      if (expanded_value_heads && decay_last <= 0) {
        fail_shape_inference("decay last dimension must be positive");
      }
      if (Dk.has_dim_value()) {
        const int64_t dk = Dk.dim_value();
        const int64_t kv_dk = checkedMultiply(state_heads, dk);
        if (decay_last != state_heads && decay_last != kv_dk) {
          fail_shape_inference(
              "decay last dim (",
              decay_last,
              ") must be the state head count (",
              state_heads,
              ") or state head count * d_k (",
              kv_dk,
              ")");
        }
      } else if (decay_last != state_heads && decay_last % state_heads != 0) {
        fail_shape_inference(
            "decay last dim (", decay_last, ") must be a multiple of the state head count (", state_heads, ")");
      }
    }
  }
  if (has_beta) {
    const auto* beta_type = ctx.getInputType(5);
    if (beta_type != nullptr && beta_type->has_tensor_type() && beta_type->tensor_type().has_shape() &&
        beta_type->tensor_type().shape().dim_size() == 3 && beta_type->tensor_type().shape().dim(2).has_dim_value()) {
      const int64_t beta_last = beta_type->tensor_type().shape().dim(2).dim_value();
      if (beta_last != 1 && beta_last != state_heads) {
        fail_shape_inference("beta last dim (", beta_last, ") must be 1 or the state head count (", state_heads, ")");
      }
    }
  }

  // Output 0: (B, T, output_heads * d_v).
  Dim OutLast;
  if (Dv.has_dim_value()) {
    OutLast.set_dim_value(checkedMultiply(output_heads, Dv.dim_value()));
  }
  updateOutputShape(ctx, 0, {B, T, OutLast});

  // Output 1: present_state (B, Hstate, d_k, d_v).
  updateOutputShape(ctx, 1, {B, Hstate, Dk, Dv});
}

static void InferLinearAttentionShape_opset27(InferenceContext& ctx) {
  InferLinearAttentionShape(ctx, false);
}

static void InferLinearAttentionShape_opset29(InferenceContext& ctx) {
  InferLinearAttentionShape(ctx, true);
}

OpSchema LinearAttentionSchema(bool expanded_value_heads) {
  // Both versions share the recurrence, types and optional-input rules.
  // Only opset 29 exposes independent value/state heads; legacy query GQA
  // remains the default. Keep the packed value, gate and output descriptions
  // aligned with that choice, while leaving Q/K packing unchanged.
  auto schema = OpSchema();
  if (expanded_value_heads) {
    schema.Attr(
        "v_num_heads",
        "Value/state head count. Zero (default) uses kv_num_heads and preserves query GQA. "
        "A positive value requires q_num_heads == kv_num_heads and must be a multiple of kv_num_heads. "
        "Q/K head h is repeated adjacently v_num_heads / kv_num_heads times. "
        "The output has v_num_heads heads, and value, state, decay and beta use this head count.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
  }
  return schema.SetDoc(expanded_value_heads ? kDoc_LinearAttention_ver29 : kDoc_LinearAttention_ver27)
      .Attr(
          "update_rule",
          "The update rule for the linear attention recurrence. "
          "One of: 'linear', 'gated', 'delta', 'gated_delta'. Default is 'gated_delta'.",
          AttributeProto::STRING,
          std::string("gated_delta"))
      .Attr(
          "scale",
          "Output scaling factor. When 0.0 (default), derives d_k = query.shape[-1] / q_num_heads "
          "and uses 1/sqrt(d_k). Set explicitly to override.",
          AttributeProto::FLOAT,
          0.0f)
      .Attr("q_num_heads", "Number of query heads. Always required.", AttributeProto::INT)
      .Attr(
          "kv_num_heads",
          expanded_value_heads ? "Number of compact key heads. Always required."
                               : "Number of key/value heads. Always required.",
          AttributeProto::INT)
      .Attr(
          "chunk_size",
          "Chunk size for the chunk-parallel WY decomposition during prefill (T>1). "
          "Tuning hint; does not affect output correctness.",
          AttributeProto::INT,
          static_cast<int64_t>(64))
      .Input(
          0,
          "query",
          "Query vectors with 3D packed shape (B, T, H_q * d_k). "
          "Heads are packed into the last dimension.",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable)
      .Input(
          1,
          "key",
          "Key vectors with 3D packed shape (B, T, H_kv * d_k). "
          "Should be L2-normalized for delta/gated_delta modes.",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable)
      .Input(
          2,
          "value",
          expanded_value_heads ? "Value vectors with 3D packed shape (B, T, H_v * d_v), where H_v is "
                                 "kv_num_heads unless a positive v_num_heads is specified."
                               : "Value vectors with 3D packed shape (B, T, H_kv * d_v).",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable)
      .Input(
          3,
          "past_state",
          expanded_value_heads ? "Recurrent state from previous step with shape (B, H_v, d_k, d_v). "
                                 "Always 4D. If not provided, defaults to zeros."
                               : "Recurrent state from previous step with shape (B, H_kv, d_k, d_v). "
                                 "Always 4D. If not provided, defaults to zeros.",
          "S",
          OpSchema::Optional,
          true,
          1,
          OpSchema::NonDifferentiable)
      .Input(
          4,
          "decay",
          expanded_value_heads ? "Exponential decay gate in log-space, with shape (B, T, H_v * d_k) "
                                 "or (B, T, H_v). Required for 'gated' and 'gated_delta' modes."
                               : "Exponential decay gate in log-space. 3D packed shape: "
                                 "(B, T, H_kv * d_k) for per-key-dimension decay (GLA/RWKV-6), or "
                                 "(B, T, H_kv) for per-head scalar decay (DeltaNet/RetNet). "
                                 "Required for 'gated' and 'gated_delta' modes.",
          "T",
          OpSchema::Optional,
          true,
          1,
          OpSchema::Differentiable)
      .Input(
          5,
          "beta",
          expanded_value_heads ? "Update rate (sigmoid output), with shape (B, T, H_v) or (B, T, 1). "
                                 "Required for 'delta' and 'gated_delta' modes."
                               : "Update rate (sigmoid output). 3D packed shape: "
                                 "(B, T, H_kv) or (B, T, 1). Required for 'delta' and 'gated_delta' modes.",
          "T",
          OpSchema::Optional,
          true,
          1,
          OpSchema::Differentiable)
      .Output(
          0,
          "output",
          expanded_value_heads ? "Attention output with 3D packed shape (B, T, H_o * d_v), where H_o is "
                                 "q_num_heads unless a positive v_num_heads is specified."
                               : "Attention output with 3D packed shape (B, T, H_q * d_v).",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable)
      .Output(
          1,
          "present_state",
          expanded_value_heads ? "Updated recurrent state with shape (B, H_v, d_k, d_v). Always 4D."
                               : "Updated recurrent state with shape (B, H_kv, d_k, d_v). Always 4D.",
          "S",
          OpSchema::Single,
          true,
          1,
          OpSchema::NonDifferentiable)
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"},
          "Constrain activation input and output types to float16, bfloat16, or float32 tensors.")
      .TypeConstraint(
          "S",
          {"tensor(float16)", "tensor(bfloat16)", "tensor(float)"},
          "Constrain state types to float16, bfloat16, or float32 tensors. "
          "Should be float32 or the same as T for numerical stability on long sequences.")
      .SetContextDependentFunctionBodyBuilder(BuildLinearAttentionFunction)
      .TypeAndShapeInferenceFunction(
          expanded_value_heads ? InferLinearAttentionShape_opset29 : InferLinearAttentionShape_opset27);
}

} // namespace ONNX_NAMESPACE::defs::nn::utils
