/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <optional>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

// ---------------------------------------------------------------------------
// FlexAttention Constants
// ---------------------------------------------------------------------------

// Expected number of inputs for modifier subgraphs: (tensor_in)
static constexpr int kModifierInputCount = 1;

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
Computes scaled dot-product attention over rank-4 (batched, multi-head) inputs,
with optional user-provided customization subgraphs at two stages:

1. score_mod: Modify the attention score tensor after Q·K^T
2. prob_mod: Modify the probability tensor after Softmax

This operator mirrors the capabilities of PyTorch's flex_attention:
https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html

Input Shapes (MUST be rank-4 tensors):
- Q: `(batch_size, q_num_heads, q_sequence_length, head_size)`
- K: `(batch_size, kv_num_heads, kv_sequence_length, head_size)`
- V: `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)`

Output Shape:
- Y: `(batch_size, q_num_heads, q_sequence_length, v_head_size)`

FlexAttention Computation:
```
Scores = (Q @ K^T) * scale
Scores = score_mod(Scores)             # if 'score_mod' is provided
Probs = Softmax(Scores, axis=-1)
Probs = prob_mod(Probs)                # if 'prob_mod' is provided
Y = Probs @ V
```

Grouped Query Attention (GQA):
When `q_num_heads != kv_num_heads`, each K/V head is shared by a contiguous
group of query heads in head-index order. Let
`group_size = q_num_heads / kv_num_heads`; then query head `h` uses K/V head
`floor(h / group_size)`. `q_num_heads` must be a multiple of
`kv_num_heads`.

Modifier Subgraphs (score_mod, prob_mod):
Each modifier subgraph takes exactly one rank-4 tensor input and must produce
exactly one rank-4 tensor output of the same shape and element type.
- score_mod input/output shape: `(batch_size, q_num_heads, q_sequence_length, kv_sequence_length)`
- prob_mod  input/output shape: `(batch_size, q_num_heads, q_sequence_length, kv_sequence_length)`
The element type is determined by softmax_precision (defaults to float32 for
non-double inputs, otherwise double).

Masking can be expressed in score_mod by writing masked positions as -inf (or a
large negative value appropriate for the target precision).
)DOC";

// ---------------------------------------------------------------------------
// Type Utilities
// ---------------------------------------------------------------------------

// Returns true if the element type supports softmax computation.
// Softmax requires floating-point types for numerical stability.
static bool IsValidSoftmaxElementType(int32_t elem_type) {
  switch (elem_type) {
    case TensorProto::FLOAT:
    case TensorProto::FLOAT16:
    case TensorProto::BFLOAT16:
    case TensorProto::DOUBLE:
      return true;
    default:
      return false;
  }
}

// Determines the element type for softmax computation.
// Priority:
//   1. Use explicit softmax_precision attribute if provided
//   2. Default to float32 for all non-double input types
//   3. Keep double as double
static int32_t GetSoftmaxElementType(int32_t input_elem_type, const AttributeProto* softmax_precision_attr) {
  // Honor explicit precision setting
  if (softmax_precision_attr != nullptr) {
    return static_cast<int32_t>(softmax_precision_attr->i());
  }
  return input_elem_type == TensorProto::DOUBLE ? TensorProto::DOUBLE : TensorProto::FLOAT;
}

// ---------------------------------------------------------------------------
// Modifier Graph Validation
// ---------------------------------------------------------------------------

// Validates a modifier graph attribute for function builder context.
// Returns true if the graph is valid or not provided.
static bool ValidateModifierGraphForBuilder(const AttributeProto* attr) {
  if (attr == nullptr) {
    return true; // Optional attribute not provided.
  }
  if (!attr->has_g()) {
    return false; // Attribute exists but is not a graph.
  }
  const auto& graph = attr->g();
  return graph.input_size() == kModifierInputCount && graph.output_size() == 1;
}

// Checks if a modifier graph is a trivial identity (passes through first input unchanged).
static bool IsIdentityModifierGraph(const AttributeProto* attr) {
  if (attr == nullptr || !attr->has_g()) {
    return false;
  }
  const auto& graph = attr->g();
  return graph.node_size() == 0 && graph.input_size() == kModifierInputCount && graph.output_size() == 1 &&
      graph.input(0).name() == graph.output(0).name();
}

// ---------------------------------------------------------------------------
// Shape Inference
// ---------------------------------------------------------------------------

// Validates a modifier graph attribute during shape inference.
// Checks I/O counts, element types, and scalar requirements.
static void ValidateModifierGraph(
    const AttributeProto* attr,
    const std::string& attr_name,
    std::optional<int32_t> expected_output_elem_type) {
  if (!attr) {
    return;
  }
  if (!attr->has_g()) {
    fail_shape_inference("Attribute ", attr_name, " must be a graph.");
  }

  const auto& graph = attr->g();

  // Validate I/O counts.
  if (graph.input_size() != kModifierInputCount) {
    fail_shape_inference(
        "Attribute ", attr_name, " expected ", kModifierInputCount, " input but graph has ", graph.input_size(), ".");
  }
  if (graph.output_size() != 1) {
    fail_shape_inference("Attribute ", attr_name, " must have exactly one output.");
  }

  // Helper: validate element type.
  auto validate_dtype = [&](const ValueInfoProto& vi, int32_t expected, const char* desc) {
    if (!vi.has_type() || !vi.type().has_tensor_type())
      return;
    if (vi.type().tensor_type().elem_type() != expected) {
      fail_shape_inference(
          "Attribute ",
          attr_name,
          " ",
          desc,
          " type mismatch. Expected ",
          expected,
          ", got ",
          vi.type().tensor_type().elem_type(),
          ".");
    }
  };

  // Validate input element types.
  if (expected_output_elem_type.has_value() && graph.input_size() >= 1) {
    validate_dtype(graph.input(0), expected_output_elem_type.value(), "input(0)");
  }

  // Validate output type and shape.
  if (graph.output(0).has_type()) {
    const auto& type = graph.output(0).type();
    if (!type.has_tensor_type()) {
      fail_shape_inference("Attribute ", attr_name, " output must be a tensor.");
    }
    const auto& tensor = type.tensor_type();
    if (expected_output_elem_type.has_value() && tensor.elem_type() != expected_output_elem_type.value()) {
      fail_shape_inference(
          "Attribute ",
          attr_name,
          " output type mismatch. Expected ",
          expected_output_elem_type.value(),
          ", got ",
          tensor.elem_type(),
          ".");
    }
  }

  // Validate modifier tensor ranks and compatibility when specified.
  if (graph.input(0).has_type() && graph.input(0).type().has_tensor_type()) {
    const auto& input_tensor = graph.input(0).type().tensor_type();
    if (input_tensor.has_shape() && input_tensor.shape().dim_size() != 4) {
      fail_shape_inference("Attribute ", attr_name, " input must be rank-4 tensor when shape is specified.");
    }
  }
  if (graph.output(0).has_type() && graph.output(0).type().has_tensor_type()) {
    const auto& output_tensor = graph.output(0).type().tensor_type();
    if (output_tensor.has_shape() && output_tensor.shape().dim_size() != 4) {
      fail_shape_inference("Attribute ", attr_name, " output must be rank-4 tensor when shape is specified.");
    }
  }
  if (graph.input(0).has_type() && graph.output(0).has_type() && graph.input(0).type().has_tensor_type() &&
      graph.output(0).type().has_tensor_type()) {
    const auto& in_shape = graph.input(0).type().tensor_type().shape();
    const auto& out_shape = graph.output(0).type().tensor_type().shape();
    const int dim_count = std::min(in_shape.dim_size(), out_shape.dim_size());
    for (int i = 0; i < dim_count; ++i) {
      if (in_shape.dim(i).has_dim_value() && out_shape.dim(i).has_dim_value() &&
          in_shape.dim(i).dim_value() != out_shape.dim(i).dim_value()) {
        fail_shape_inference("Attribute ", attr_name, " output shape must match input shape when specified.");
      }
    }
  }
}

static void FlexAttentionShapeInference(InferenceContext& ctx) {
  // Require Q, K, V inputs.
  if (ctx.getNumInputs() < 3) {
    fail_type_inference("FlexAttention requires query, key, and value inputs.");
  }

  const auto* q_type = ctx.getInputType(0);
  const auto* k_type = ctx.getInputType(1);
  const auto* v_type = ctx.getInputType(2);
  if (!q_type || !k_type || !v_type) {
    return;
  }
  if (!q_type->has_tensor_type() || !k_type->has_tensor_type() || !v_type->has_tensor_type()) {
    fail_type_inference("Inputs 0, 1, and 2 are required tensor types.");
  }

  // Validate element type consistency.
  const auto q_elem_type = q_type->tensor_type().elem_type();
  const auto k_elem_type = k_type->tensor_type().elem_type();
  const auto v_elem_type = v_type->tensor_type().elem_type();
  if (q_elem_type != k_elem_type || q_elem_type != v_elem_type) {
    fail_type_inference("Inputs query, key, and value must have the same element type.");
  }

  // Set output element type.
  auto* output_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_type->set_elem_type(q_elem_type);

  // Validate input ranks.
  const auto& q_shape = q_type->tensor_type().shape();
  const auto& k_shape = k_type->tensor_type().shape();
  const auto& v_shape = v_type->tensor_type().shape();
  if (q_shape.dim_size() != 4 || k_shape.dim_size() != 4 || v_shape.dim_size() != 4) {
    fail_shape_inference(
        "FlexAttention requires rank-4 inputs: "
        "Q (B, Hq, L, Dqk), K (B, Hkv, S, Dqk), V (B, Hkv, S, Dv).");
  }

  // Validate K and V share the same number of heads.
  if (k_shape.dim(1).has_dim_value() && v_shape.dim(1).has_dim_value() &&
      k_shape.dim(1).dim_value() != v_shape.dim(1).dim_value()) {
    fail_shape_inference("Key and value must share the same head dimension.");
  }

  // Validate K and V share the same sequence length.
  if (k_shape.dim(2).has_dim_value() && v_shape.dim(2).has_dim_value() &&
      k_shape.dim(2).dim_value() != v_shape.dim(2).dim_value()) {
    fail_shape_inference("Key and value must share the same sequence length.");
  }

  // Validate Q and K share the same embedding dimension.
  if (q_shape.dim(3).has_dim_value() && k_shape.dim(3).has_dim_value() &&
      q_shape.dim(3).dim_value() != k_shape.dim(3).dim_value()) {
    fail_shape_inference("Query and key must share the same embedding dimension.");
  }

  // Validate Grouped Query Attention (GQA) configuration.
  if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value()) {
    const auto hq = q_shape.dim(1).dim_value();
    const auto hkv = k_shape.dim(1).dim_value();
    if (hq != hkv && (hkv <= 0 || (hq % hkv) != 0)) {
      fail_shape_inference("q_num_heads must be a multiple of kv_num_heads when they differ.");
    }
  }

  // Set output shape: (B, Hq, L, Dv).
  auto* output_shape = output_type->mutable_shape();
  output_shape->clear_dim();
  *output_shape->add_dim() = q_shape.dim(0); // Batch
  *output_shape->add_dim() = q_shape.dim(1); // Query heads
  *output_shape->add_dim() = q_shape.dim(2); // Query sequence length
  *output_shape->add_dim() = v_shape.dim(3); // Value head size

  // Merge batch dimension info from all inputs.
  mergeInDimensionInfo(k_shape.dim(0), *output_shape->mutable_dim(0), 0);
  mergeInDimensionInfo(v_shape.dim(0), *output_shape->mutable_dim(0), 0);
  if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value() &&
      q_shape.dim(1).dim_value() == k_shape.dim(1).dim_value()) {
    mergeInDimensionInfo(k_shape.dim(1), *output_shape->mutable_dim(1), 1);
  }

  // Validate softmax precision.
  const int32_t softmax_elem_type = GetSoftmaxElementType(q_elem_type, ctx.getAttribute("softmax_precision"));
  if (!IsValidSoftmaxElementType(softmax_elem_type)) {
    fail_type_inference("softmax_precision must be specified when inputs are not float/float16/bfloat16/double.");
  }

  // Validate modifier graphs.
  ValidateModifierGraph(ctx.getAttribute("score_mod"), "score_mod", softmax_elem_type);
  ValidateModifierGraph(ctx.getAttribute("prob_mod"), "prob_mod", softmax_elem_type);
}

// ---------------------------------------------------------------------------
// Function Body Builder
// ---------------------------------------------------------------------------

// Builds the function body for FlexAttention.
// The following pattern is applied:
//
//      Q          K          V
//      |          |          |
//      |       Transpose     |
//      |          |          |
//      ---MatMul---          |
//            |               |
//         * scale            |
//            |               |
//      [score_mod]           |
//            |               |
//         Softmax            |
//            |               |
//       [prob_mod]           |
//            |               |
//            -----MatMul------
//                   |
//                   Y
//
static bool BuildFlexAttentionFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // Validate and extract input types
  const auto* q_type = ctx.getInputType(0);
  const auto* k_type = ctx.getInputType(1);
  const auto* v_type = ctx.getInputType(2);

  if (!q_type || !k_type || !v_type || !q_type->has_tensor_type() || !k_type->has_tensor_type() ||
      !v_type->has_tensor_type()) {
    return false;
  }

  const int64_t input_elem_type = q_type->tensor_type().elem_type();
  if (k_type->tensor_type().elem_type() != input_elem_type || v_type->tensor_type().elem_type() != input_elem_type) {
    return false;
  }

  // Determine softmax precision
  const int32_t softmax_precision =
      GetSoftmaxElementType(static_cast<int32_t>(input_elem_type), ctx.getAttribute("softmax_precision"));
  if (!IsValidSoftmaxElementType(softmax_precision)) {
    return false;
  }

  const auto* scale_attr = ctx.getAttribute("scale");
  const float scale = scale_attr ? scale_attr->f() : 1.0f;

  // Validate modifier graphs
  const auto* score_mod_attr = ctx.getAttribute("score_mod");
  const auto* prob_mod_attr = ctx.getAttribute("prob_mod");

  if (!ValidateModifierGraphForBuilder(score_mod_attr) || !ValidateModifierGraphForBuilder(prob_mod_attr)) {
    return false;
  }

  // Check which modifier graphs are needed.
  const bool need_score_mod = score_mod_attr && !IsIdentityModifierGraph(score_mod_attr);
  const bool need_prob_mod = prob_mod_attr && !IsIdentityModifierGraph(prob_mod_attr);

  // Build the function graph
  constexpr int64_t kFloat32 = TensorProto_DataType_FLOAT;

  FunctionBuilder builder(functionProto);

  // Extract shape information.
  builder.Add("BatchSize = Shape <start = 0, end = 1> (Q)").Add("KVSeqLen = Shape <start = -2, end = -1> (K)");

  builder.Add("QNumHeads = Shape <start = 1, end = 2> (Q)").Add("KVNumHeads = Shape <start = 1, end = 2> (K)");

  // Handle Grouped Query Attention (GQA) by sharing each K/V head across
  // contiguous query-head groups when needed.
  builder.Add("KVRepeat = Div (QNumHeads, KVNumHeads)")
      .Const1D("Axis2", static_cast<int64_t>(2))
      // K: [B, Hkv, S, E] -> [B, Hq, S, E]
      .Add("KUnsqueezed = Unsqueeze(K, Axis2)")
      .Add("KHeadSize1D = Shape <start = 3, end = 4> (K)")
      .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, KVRepeat, KVSeqLen, KHeadSize1D)")
      .Add("KExpanded = Expand(KUnsqueezed, KExpandShape)")
      .Add("KAlignedShape = Concat <axis = 0> (BatchSize, QNumHeads, KVSeqLen, KHeadSize1D)")
      .Add("KAligned = Reshape(KExpanded, KAlignedShape)")
      // V: same transformation
      .Add("VUnsqueezed = Unsqueeze(V, Axis2)")
      .Add("VHeadSize1D = Shape <start = 3, end = 4> (V)")
      .Add("VExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, KVRepeat, KVSeqLen, VHeadSize1D)")
      .Add("VExpanded = Expand(VUnsqueezed, VExpandShape)")
      .Add("VAlignedShape = Concat <axis = 0> (BatchSize, QNumHeads, KVSeqLen, VHeadSize1D)")
      .Add("VAligned = Reshape(VExpanded, VAlignedShape)");

  // Compute scaling factor.
  builder.Add("QShapeAll = Shape(Q)")
      .Const("Idx3Head", ToTensor<int64_t>(3))
      .Add("QKHeadSize = Gather <axis = 0> (QShapeAll, Idx3Head)")
      .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", kFloat32)
      .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
      .Const("OneF", ToTensor<float>(1.0f))
      .Add("CalculatedScale = Div(OneF, SqrtHeadSize)")
      .Const("ScaleF", ToTensor<float>(scale));

  if (scale_attr != nullptr) {
    builder.Add("ScaleFactorF32 = Identity(ScaleF)");
  } else {
    builder.Add("ScaleFactorF32 = Identity(CalculatedScale)");
  }
  builder.Add("ScaleFactorSqrtF32 = Sqrt(ScaleFactorF32)");

  // Compute attention scores by pre-scaling Q and K with sqrt(scale).
  builder.Add("QF = Cast (Q)", "to", kFloat32)
      .Add("KF = Cast (KAligned)", "to", kFloat32)
      .Add("QScaledF = Mul(QF, ScaleFactorSqrtF32)")
      .Add("KScaledF = Mul(KF, ScaleFactorSqrtF32)")
      .Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KScaledF)")
      .Add("ScoreF = MatMul(QScaledF, KTranspose)")
      .Add("SoftmaxCast = Cast (ScoreF)", "to", softmax_precision);

  // Apply score_mod.
  if (need_score_mod) {
    builder.AddInlinedCall({"ScoreAfterScoreMod"}, score_mod_attr->g(), {"SoftmaxCast"}, "ScoreMod_");
  } else {
    builder.Add("ScoreAfterScoreMod = Identity(SoftmaxCast)");
  }

  // Apply Softmax.
  builder.Add("Prob = Softmax <axis = 3> (ScoreAfterScoreMod)");

  // Apply prob_mod.
  if (need_prob_mod) {
    builder.AddInlinedCall({"ProbAfterProbMod"}, prob_mod_attr->g(), {"Prob"}, "ProbMod_");
  } else {
    builder.Add("ProbAfterProbMod = Identity(Prob)");
  }

  // Compute final output: Prob @ V.
  if (static_cast<int64_t>(softmax_precision) != input_elem_type) {
    builder.Add("VSp = Cast (VAligned)", "to", softmax_precision)
        .Add("YSp = MatMul (ProbAfterProbMod, VSp)")
        .Add("Y = Cast (YSp)", "to", input_elem_type);
  } else {
    builder.Add("Y = MatMul (ProbAfterProbMod, VAligned)");
  }

  // Finalize function.
  schema.BuildFunction(functionProto);

  return true;
}

// ---------------------------------------------------------------------------
// Operator Schema Registration
// ---------------------------------------------------------------------------

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
        .Input(0, "Q", "Query tensor with shape `(batch_size, q_num_heads, q_seq_len, head_size)`.", "T1")
        .Input(1, "K", "Key tensor with shape `(batch_size, kv_num_heads, kv_seq_len, head_size)`.", "T1")
        .Input(2, "V", "Value tensor with shape `(batch_size, kv_num_heads, kv_seq_len, v_head_size)`.", "T1")
        .Output(0, "Y", "Output tensor with shape `(batch_size, q_num_heads, q_seq_len, v_head_size)`.", "T1")
        .Attr(
            "scale",
            "Scaling factor for Q*K^T. Defaults to 1/sqrt(head_size).",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "softmax_precision",
            "Floating-point precision for softmax computation. "
            "Defaults to float32 for non-double inputs, otherwise uses double. "
            "Must be explicitly specified for non-float types.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "score_mod",
            "Optional score modifier subgraph with 1 rank-4 tensor input and 1 rank-4 tensor output "
            "of the same shape and element type: (scores) -> scores_out. "
            "scores has softmax_precision element type and shape (B, Hq, L, S). "
            "The output must preserve the input shape.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "prob_mod",
            "Optional probability modifier subgraph with 1 rank-4 tensor input and 1 rank-4 tensor output "
            "of the same shape and element type: (probs) -> probs_out. "
            "probs has softmax_precision element type and shape (B, Hq, L, S). "
            "The output must preserve the input shape.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q, K, V to float tensors.")
        .TypeAndShapeInferenceFunction(FlexAttentionShapeInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(BuildFlexAttentionFunctionBody));

} // namespace ONNX_NAMESPACE
