/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_set>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

// ---------------------------------------------------------------------------
// FlexAttention Constants
// ---------------------------------------------------------------------------

// Expected number of inputs for score/prob modifier subgraphs:
// (score_or_prob, batch_idx, head_idx, q_idx, k_idx)
static constexpr int kScoreProbModInputCount = 5;

// Expected number of inputs for mask modifier subgraph:
// (batch_idx, head_idx, q_idx, k_idx)
static constexpr int kMaskModInputCount = 4;

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
Computes scaled dot-product attention over rank-4 (batched, multi-head) inputs,
with optional user-provided customization subgraphs at up to three stages:

1. score_mod: Modify each scalar attention score after Q·K^T
2. mask_mod: Determine which (q_idx, k_idx) connections are allowed
3. prob_mod: Modify each scalar probability after Softmax

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
Scores = score_mod(Scores)             # if provided
Scores = apply_mask(Scores, mask_mod)  # if provided, masked positions get -inf
Probs = Softmax(Scores, axis=-1)
Probs = prob_mod(Probs)                # if provided
Y = Probs @ V
```

Grouped Query Attention (GQA):
When `enable_gqa=1`, supports GQA where `q_num_heads` is a multiple of `kv_num_heads`.
K/V heads are broadcast to match query heads count.

Note: The default function body uses a Loop for element-wise modifier
application, which is intended as a fallback. Optimized backends should
recognize this pattern and apply fused kernel implementations.
)DOC";

// ---------------------------------------------------------------------------
// Forward Declarations
// ---------------------------------------------------------------------------

static void RemapGraphProtoNames(GraphProto* g, const std::function<std::string(const std::string&)>& map_name);

// ---------------------------------------------------------------------------
// Graph Utilities
// ---------------------------------------------------------------------------

// Finds the last node index that produces any of the specified values.
// Returns -1 if no producer is found.
static int FindLastProducerIndex(const FunctionProto& fp, const std::vector<std::string>& values) {
  std::unordered_set<std::string> value_set(values.begin(), values.end());
  int last_index = -1;
  for (int i = 0; i < fp.node_size(); ++i) {
    for (const auto& output : fp.node(i).output()) {
      if (value_set.count(output)) {
        last_index = std::max(last_index, i);
        break;
      }
    }
  }
  return last_index;
}

// Inserts a node at the specified position in the function's node list.
static void InsertNodeAt(FunctionProto& fp, const NodeProto& node, int index) {
  auto* nodes = fp.mutable_node();
  nodes->Add()->CopyFrom(node);
  // Bubble the new node from the end to the desired position.
  for (int i = nodes->size() - 1; i > index; --i) {
    nodes->SwapElements(i, i - 1);
  }
}

// Remaps input/output names in a NodeProto using the provided mapping function.
static void RemapNodeProtoNames(NodeProto* node, const std::function<std::string(const std::string&)>& map_name) {
  // Store and clear original inputs/outputs.
  const auto old_inputs = node->input();
  const auto old_outputs = node->output();
  node->clear_input();
  node->clear_output();

  // Apply name mapping.
  for (const auto& input : old_inputs) {
    node->add_input(map_name(input));
  }
  for (const auto& output : old_outputs) {
    node->add_output(map_name(output));
  }

  // Recursively remap nested subgraphs (If/Loop/Scan/etc.).
  for (int i = 0; i < node->attribute_size(); ++i) {
    auto* attr = node->mutable_attribute(i);
    if (attr->has_g()) {
      RemapGraphProtoNames(attr->mutable_g(), map_name);
    }
    for (int j = 0; j < attr->graphs_size(); ++j) {
      RemapGraphProtoNames(attr->mutable_graphs(j), map_name);
    }
  }
}

// Remaps all value names in a GraphProto using the provided mapping function.
static void RemapGraphProtoNames(GraphProto* graph, const std::function<std::string(const std::string&)>& map_name) {
  // Remap graph I/O.
  for (int i = 0; i < graph->input_size(); ++i) {
    graph->mutable_input(i)->set_name(map_name(graph->input(i).name()));
  }
  for (int i = 0; i < graph->output_size(); ++i) {
    graph->mutable_output(i)->set_name(map_name(graph->output(i).name()));
  }

  // Remap value_info entries.
  for (int i = 0; i < graph->value_info_size(); ++i) {
    graph->mutable_value_info(i)->set_name(map_name(graph->value_info(i).name()));
  }

  // Remap initializers.
  for (int i = 0; i < graph->initializer_size(); ++i) {
    graph->mutable_initializer(i)->set_name(map_name(graph->initializer(i).name()));
  }
  for (int i = 0; i < graph->sparse_initializer_size(); ++i) {
    auto* sparse = graph->mutable_sparse_initializer(i);
    if (sparse->has_values()) {
      sparse->mutable_values()->set_name(map_name(sparse->values().name()));
    }
    if (sparse->has_indices()) {
      sparse->mutable_indices()->set_name(map_name(sparse->indices().name()));
    }
  }

  // Remap all nodes.
  for (int i = 0; i < graph->node_size(); ++i) {
    RemapNodeProtoNames(graph->mutable_node(i), map_name);
  }
}

// Inlines a subgraph into a destination graph with name remapping.
// Maps subgraph I/O names to caller-provided names using io_map.
// Prefixes all other value names with the provided prefix to avoid collisions.
static void InlineSubgraphInto(
    GraphProto* dst,
    const GraphProto& src,
    const std::unordered_map<std::string, std::string>& io_map,
    const std::string& prefix) {
  auto map_name = [&](const std::string& name) -> std::string {
    if (name.empty()) {
      return name;
    }
    auto it = io_map.find(name);
    return (it != io_map.end()) ? it->second : (prefix + name);
  };

  // Copy initializers with remapped names.
  for (const auto& init : src.initializer()) {
    TensorProto* new_init = dst->add_initializer();
    *new_init = init;
    new_init->set_name(map_name(init.name()));
  }
  for (const auto& sparse_init : src.sparse_initializer()) {
    SparseTensorProto* new_sparse = dst->add_sparse_initializer();
    *new_sparse = sparse_init;
    if (new_sparse->has_values()) {
      new_sparse->mutable_values()->set_name(map_name(new_sparse->values().name()));
    }
    if (new_sparse->has_indices()) {
      new_sparse->mutable_indices()->set_name(map_name(new_sparse->indices().name()));
    }
  }

  // Copy nodes with remapped I/O names.
  for (const auto& node : src.node()) {
    NodeProto* new_node = dst->add_node();
    *new_node = node;
    RemapNodeProtoNames(new_node, map_name);
  }
}

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
//   2. Promote float16/bfloat16 to float32 for numerical stability
//   3. Use input type as-is for float32/double
static int32_t GetSoftmaxElementType(int32_t input_elem_type, const AttributeProto* softmax_precision_attr) {
  // Honor explicit precision setting
  if (softmax_precision_attr != nullptr) {
    return static_cast<int32_t>(softmax_precision_attr->i());
  }
  // Auto-promote half-precision types to float32
  if (input_elem_type == TensorProto::FLOAT16 || input_elem_type == TensorProto::BFLOAT16) {
    return TensorProto::FLOAT;
  }
  return input_elem_type;
}

// ---------------------------------------------------------------------------
// Modifier Graph Validation
// ---------------------------------------------------------------------------

// Validates a modifier graph attribute for function builder context.
// Returns true if the graph is valid or not provided.
static bool ValidateModifierGraphForBuilder(const AttributeProto* attr, int expected_inputs) {
  if (attr == nullptr) {
    return true; // Optional attribute not provided.
  }
  if (!attr->has_g()) {
    return false; // Attribute exists but is not a graph.
  }
  const auto& graph = attr->g();
  return graph.input_size() == expected_inputs && graph.output_size() == 1;
}

// Checks if a modifier graph is a trivial identity (passes through first input unchanged).
static bool IsIdentityModifierGraph(const AttributeProto* attr, int expected_inputs) {
  if (attr == nullptr || !attr->has_g()) {
    return false;
  }
  const auto& graph = attr->g();
  return graph.node_size() == 0 && graph.input_size() == expected_inputs && graph.output_size() == 1 &&
      graph.input(0).name() == graph.output(0).name();
}

// ---------------------------------------------------------------------------
// Loop Body Graph Builders
// ---------------------------------------------------------------------------

// Adds a binary operation node (e.g., Add, Mul, Div, Mod) to a graph.
static void
AddBinaryNode(GraphProto* graph, const char* op_type, const char* lhs, const char* rhs, const char* output) {
  auto* node = graph->add_node();
  node->set_op_type(op_type);
  node->add_input(lhs);
  node->add_input(rhs);
  node->add_output(output);
}

// Adds an Identity node to a graph.
static void AddIdentityNode(GraphProto* graph, const char* input, const char* output) {
  auto* node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input(input);
  node->add_output(output);
}

// Adds a Gather node with axis=0 to a graph.
static void AddGatherNode(GraphProto* graph, const char* data, const char* indices, const char* output) {
  auto* node = graph->add_node();
  node->set_op_type("Gather");
  node->add_input(data);
  node->add_input(indices);
  node->add_output(output);
  auto* axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::INT);
  axis_attr->set_i(0);
}

// Adds index decomposition nodes for scalar-contract Loop body.
// Computes (batch, head, q_idx, k_idx) from a flattened iteration index.
// Index layout for [B, H, L, S] tensor (row-major):
//   k_idx = iter % S,  q_idx = (iter / S) % L
//   head  = (iter / S / L) % H,  batch = (iter / S / L) / H
static void AddLoopIndexDecompositionNodes(GraphProto* body) {
  AddBinaryNode(body, "Mod", "iter", "S", "k_idx");
  AddBinaryNode(body, "Div", "iter", "S", "t1");
  AddBinaryNode(body, "Mod", "t1", "L", "q_idx");
  AddBinaryNode(body, "Div", "t1", "L", "t2");
  AddBinaryNode(body, "Mod", "t2", "H", "head");
  AddBinaryNode(body, "Div", "t2", "H", "batch");
}

// Creates a Loop node for the scalar-contract pattern.
// Returns a pointer to the body GraphProto for further customization.
static GraphProto* CreateScalarContractLoopNode(NodeProto& loop_node, const char* output_name, const char* body_name) {
  loop_node.Clear();
  loop_node.set_op_type("Loop");
  loop_node.add_input("N");
  loop_node.add_input("CondInit");
  loop_node.add_output(output_name);

  auto* body_attr = loop_node.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(AttributeProto::GRAPH);

  GraphProto* body = body_attr->mutable_g();
  body->set_name(body_name);
  body->add_input()->set_name("iter"); // INT64 scalar (loop counter)
  body->add_input()->set_name("cond_in"); // BOOL scalar (continue condition)
  body->add_output()->set_name("cond_out"); // BOOL scalar (next condition)
  body->add_output()->set_name("scan_out"); // Output value to accumulate

  return body;
}

// Finalizes a scalar-contract Loop body with termination nodes.
static void FinalizeScalarContractLoopBody(GraphProto* body, const char* result_name) {
  AddIdentityNode(body, "cond_in", "cond_out");
  AddIdentityNode(body, result_name, "scan_out");
}

// ---------------------------------------------------------------------------
// Shape Inference
// ---------------------------------------------------------------------------

// Validates a modifier graph attribute during shape inference.
// Checks I/O counts, element types, and scalar requirements.
static void ValidateModifierGraph(
    const AttributeProto* attr,
    size_t expected_inputs,
    const std::string& attr_name,
    std::optional<int32_t> expected_output_elem_type,
    bool require_scalar_output) {
  if (!attr) {
    return;
  }
  if (!attr->has_g()) {
    fail_shape_inference("Attribute ", attr_name, " must be a graph.");
  }

  const auto& graph = attr->g();

  // Validate I/O counts.
  if (graph.input_size() != static_cast<int>(expected_inputs)) {
    fail_shape_inference(
        "Attribute ", attr_name, " expected ", expected_inputs, " inputs but graph has ", graph.input_size(), ".");
  }
  if (graph.output_size() != 1) {
    fail_shape_inference("Attribute ", attr_name, " must have exactly one output.");
  }

  // Helper: validate scalar tensor shape.
  auto validate_scalar = [&](const ValueInfoProto& vi, const char* desc) {
    if (!vi.has_type())
      return;
    if (!vi.type().has_tensor_type()) {
      fail_shape_inference("Attribute ", attr_name, " ", desc, " must be a tensor.");
    }
    const auto& tensor = vi.type().tensor_type();
    if (tensor.has_shape() && tensor.shape().dim_size() != 0) {
      fail_shape_inference("Attribute ", attr_name, " ", desc, " must be a scalar (0-D).");
    }
  };

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

  const bool is_mask_mod = (expected_inputs == kMaskModInputCount);

  // Validate all inputs are scalar tensors.
  for (int i = 0; i < graph.input_size(); ++i) {
    std::string desc = "input(" + std::to_string(i) + ")";
    validate_scalar(graph.input(i), desc.c_str());
  }

  // Validate input element types.
  if (is_mask_mod) {
    // mask_mod: all inputs are INT64 indices.
    for (int i = 0; i < graph.input_size(); ++i) {
      std::string desc = "input(" + std::to_string(i) + ")";
      validate_dtype(graph.input(i), TensorProto::INT64, desc.c_str());
    }
  } else if (expected_output_elem_type.has_value()) {
    // score_mod/prob_mod: first input matches output type, rest are INT64.
    if (graph.input_size() >= 1) {
      validate_dtype(graph.input(0), expected_output_elem_type.value(), "input(0)");
    }
    for (int i = 1; i < graph.input_size(); ++i) {
      std::string desc = "input(" + std::to_string(i) + ")";
      validate_dtype(graph.input(i), TensorProto::INT64, desc.c_str());
    }
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
    if (require_scalar_output && tensor.has_shape() && tensor.shape().dim_size() != 0) {
      fail_shape_inference("Attribute ", attr_name, " output must be scalar.");
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
  const auto enable_gqa = getAttribute(ctx, "enable_gqa", 0);
  if (enable_gqa == 0) {
    // Standard attention: require Hq == Hkv.
    if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value() &&
        q_shape.dim(1).dim_value() != k_shape.dim(1).dim_value()) {
      fail_shape_inference("enable_gqa=0 requires Hq == Hkv.");
    }
  } else {
    // GQA: require Hq to be divisible by Hkv.
    if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value()) {
      const auto hq = q_shape.dim(1).dim_value();
      const auto hkv = k_shape.dim(1).dim_value();
      if (hkv <= 0 || (hq % hkv) != 0) {
        fail_shape_inference("enable_gqa=1 requires Hq to be divisible by Hkv.");
      }
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
  if (enable_gqa == 0) {
    mergeInDimensionInfo(k_shape.dim(1), *output_shape->mutable_dim(1), 1);
  }

  // Validate softmax precision.
  const int32_t softmax_elem_type = GetSoftmaxElementType(q_elem_type, ctx.getAttribute("softmax_precision"));
  if (!IsValidSoftmaxElementType(softmax_elem_type)) {
    fail_type_inference("softmax_precision must be specified when inputs are not float/float16/bfloat16/double.");
  }

  // Validate modifier graphs.
  ValidateModifierGraph(ctx.getAttribute("score_mod"), kScoreProbModInputCount, "score_mod", softmax_elem_type, true);
  ValidateModifierGraph(ctx.getAttribute("mask_mod"), kMaskModInputCount, "mask_mod", TensorProto::BOOL, true);
  ValidateModifierGraph(ctx.getAttribute("prob_mod"), kScoreProbModInputCount, "prob_mod", softmax_elem_type, true);
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
//      [mask_mod]            |
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

  // Extract configuration attributes
  const auto* enable_gqa_attr = ctx.getAttribute("enable_gqa");
  const bool enable_gqa = (enable_gqa_attr && enable_gqa_attr->i() != 0);

  const auto* scale_attr = ctx.getAttribute("scale");
  const float scale = scale_attr ? scale_attr->f() : 1.0f;

  // Use -infinity as default mask value (same pattern as Attention operator)
  const auto* mask_value_attr = ctx.getAttribute("mask_value");
  const float mask_value = mask_value_attr ? mask_value_attr->f() : -std::numeric_limits<float>::infinity();

  // Validate modifier graphs
  const auto* score_mod_attr = ctx.getAttribute("score_mod");
  const auto* mask_mod_attr = ctx.getAttribute("mask_mod");
  const auto* prob_mod_attr = ctx.getAttribute("prob_mod");

  if (!ValidateModifierGraphForBuilder(score_mod_attr, kScoreProbModInputCount) ||
      !ValidateModifierGraphForBuilder(mask_mod_attr, kMaskModInputCount) ||
      !ValidateModifierGraphForBuilder(prob_mod_attr, kScoreProbModInputCount)) {
    return false;
  }

  // Check which modifier loops are needed.
  const bool need_score_loop = score_mod_attr && !IsIdentityModifierGraph(score_mod_attr, kScoreProbModInputCount);
  const bool need_mask_loop = (mask_mod_attr != nullptr);
  const bool need_prob_loop = prob_mod_attr && !IsIdentityModifierGraph(prob_mod_attr, kScoreProbModInputCount);

  // Build the function graph
  constexpr int64_t kFloat32 = TensorProto_DataType_FLOAT;

  NodeProto score_loop_node, mask_loop_node, prob_loop_node;
  FunctionBuilder builder(functionProto);

  // Extract shape information.
  builder.Add("BatchSize = Shape <start = 0, end = 1> (Q)")
      .Add("QSeqLen = Shape <start = -2, end = -1> (Q)")
      .Add("KVSeqLen = Shape <start = -2, end = -1> (K)");

  // Input passthrough (already in [B, H, L, D] format).
  builder.Add("QReshaped = Identity(Q)")
      .Add("KReshaped = Identity(K)")
      .Add("VReshaped = Identity(V)")
      .Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)")
      .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)");

  // Handle Grouped Query Attention (GQA).
  if (enable_gqa) {
    builder.Add("KVRepeat = Div (QNumHeads, KVNumHeads)")
        .Const1D("Axis2", static_cast<int64_t>(2))
        // K: [B, Hkv, S, E] -> [B, Hq, S, E]
        .Add("KUnsqueezed = Unsqueeze(KReshaped, Axis2)")
        .Add("KHeadSize1D = Shape <start = 3, end = 4> (KReshaped)")
        .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, KVRepeat, KVSeqLen, KHeadSize1D)")
        .Add("KExpanded = Expand(KUnsqueezed, KExpandShape)")
        .Add("KAlignedShape = Concat <axis = 0> (BatchSize, QNumHeads, KVSeqLen, KHeadSize1D)")
        .Add("KAligned = Reshape(KExpanded, KAlignedShape)")
        // V: same transformation
        .Add("VUnsqueezed = Unsqueeze(VReshaped, Axis2)")
        .Add("VHeadSize1D = Shape <start = 3, end = 4> (VReshaped)")
        .Add("VExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, KVRepeat, KVSeqLen, VHeadSize1D)")
        .Add("VExpanded = Expand(VUnsqueezed, VExpandShape)")
        .Add("VAlignedShape = Concat <axis = 0> (BatchSize, QNumHeads, KVSeqLen, VHeadSize1D)")
        .Add("VAligned = Reshape(VExpanded, VAlignedShape)");
  } else {
    builder.Add("KAligned = Identity(KReshaped)").Add("VAligned = Identity(VReshaped)");
  }

  // Compute scaling factor.
  builder.Add("QShapeAll = Shape(QReshaped)")
      .Const("Idx3Head", ToTensor<int64_t>(3))
      .Add("QKHeadSize = Gather <axis = 0> (QShapeAll, Idx3Head)")
      .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", kFloat32)
      .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
      .Const1D("NegOne1D", static_cast<int64_t>(-1))
      .Const("OneF", ToTensor<float>(1.0f))
      .Add("CalculatedScale = Div(OneF, SqrtHeadSize)")
      .Const("ScaleF", ToTensor<float>(scale))
      .Add(scale_attr != nullptr ? "ScaleFactorF32 = Identity(ScaleF)" : "ScaleFactorF32 = Identity(CalculatedScale)");

  // Compute attention scores: (Q @ K^T) * scale.
  builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAligned)")
      .Add("Score = MatMul(QReshaped, KTranspose)")
      .Add("ScoreF = Cast (Score)", "to", kFloat32)
      .Add("SoftmaxCastF = Mul(ScoreF, ScaleFactorF32)")
      .Add("SoftmaxCast = Cast (SoftmaxCastF)", "to", softmax_precision);

  // Build common scalars for modifier loops.
  if (need_score_loop || need_mask_loop || need_prob_loop) {
    builder.Add("ScoreShape = Shape(SoftmaxCast)")
        .Const("Idx0", ToTensor<int64_t>(0))
        .Const("Idx1", ToTensor<int64_t>(1))
        .Const("Idx2", ToTensor<int64_t>(2))
        .Const("Idx3", ToTensor<int64_t>(3))
        .Add("B = Gather <axis = 0> (ScoreShape, Idx0)")
        .Add("H = Gather <axis = 0> (ScoreShape, Idx1)")
        .Add("L = Gather <axis = 0> (ScoreShape, Idx2)")
        .Add("S = Gather <axis = 0> (ScoreShape, Idx3)")
        .Add("ScoreFlat = Reshape(SoftmaxCast, NegOne1D)")
        .Add("N = Size(ScoreFlat)")
        .Const("TrueI64", ToTensor<int64_t>(1))
        .Add("CondInit = Cast(TrueI64)", "to", static_cast<int64_t>(TensorProto::BOOL));
  }

  // Apply score_mod.
  if (need_score_loop) {
    const auto& graph = score_mod_attr->g();
    GraphProto* body = CreateScalarContractLoopNode(score_loop_node, "ScoreModFlat", "FlexAttention_score_mod_body");

    AddLoopIndexDecompositionNodes(body);
    AddGatherNode(body, "ScoreFlat", "iter", "score_i");

    std::unordered_map<std::string, std::string> io_map{
        {graph.input(0).name(), "score_i"},
        {graph.input(1).name(), "batch"},
        {graph.input(2).name(), "head"},
        {graph.input(3).name(), "q_idx"},
        {graph.input(4).name(), "k_idx"},
        {graph.output(0).name(), "score_mod_out"}};
    InlineSubgraphInto(body, graph, io_map, "SM_");
    FinalizeScalarContractLoopBody(body, "score_mod_out");

    builder.Add("ScoreAfterScoreMod = Reshape(ScoreModFlat, ScoreShape)");
  } else {
    builder.Add("ScoreAfterScoreMod = Identity(SoftmaxCast)");
  }

  // Apply mask_mod.
  if (need_mask_loop) {
    const auto& graph = mask_mod_attr->g();
    GraphProto* body = CreateScalarContractLoopNode(mask_loop_node, "MaskFlat", "FlexAttention_mask_mod_body");

    AddLoopIndexDecompositionNodes(body);

    std::unordered_map<std::string, std::string> io_map{
        {graph.input(0).name(), "batch"},
        {graph.input(1).name(), "head"},
        {graph.input(2).name(), "q_idx"},
        {graph.input(3).name(), "k_idx"},
        {graph.output(0).name(), "mask_mod_out"}};
    InlineSubgraphInto(body, graph, io_map, "MM_");
    FinalizeScalarContractLoopBody(body, "mask_mod_out");

    builder.Add("Mask = Reshape(MaskFlat, ScoreShape)")
        .Const("MaskValueF", ToTensor<float>(mask_value))
        .Add("MaskValueSp = Cast (MaskValueF)", "to", softmax_precision)
        .Add("ScoreAfterMask = Where (Mask, ScoreAfterScoreMod, MaskValueSp)");
  } else {
    builder.Add("ScoreAfterMask = Identity(ScoreAfterScoreMod)");
  }

  // Apply Softmax.
  builder.Add("Prob = Softmax <axis = 3> (ScoreAfterMask)");

  // Apply prob_mod.
  if (need_prob_loop) {
    const auto& graph = prob_mod_attr->g();
    builder.Add("ProbFlat = Reshape(Prob, NegOne1D)");

    GraphProto* body = CreateScalarContractLoopNode(prob_loop_node, "ProbModFlat", "FlexAttention_prob_mod_body");

    AddLoopIndexDecompositionNodes(body);
    AddGatherNode(body, "ProbFlat", "iter", "prob_i");

    std::unordered_map<std::string, std::string> io_map{
        {graph.input(0).name(), "prob_i"},
        {graph.input(1).name(), "batch"},
        {graph.input(2).name(), "head"},
        {graph.input(3).name(), "q_idx"},
        {graph.input(4).name(), "k_idx"},
        {graph.output(0).name(), "prob_mod_out"}};
    InlineSubgraphInto(body, graph, io_map, "PM_");
    FinalizeScalarContractLoopBody(body, "prob_mod_out");

    builder.Add("ProbAfterProbMod = Reshape(ProbModFlat, ScoreShape)");
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

  // Finalize function and insert loop nodes.
  schema.BuildFunction(functionProto);

  const std::vector<std::string> loop_deps = {"ScoreFlat", "N", "CondInit", "B", "H", "L", "S"};

  if (need_score_loop) {
    int pos = FindLastProducerIndex(functionProto, loop_deps) + 1;
    InsertNodeAt(functionProto, score_loop_node, std::max(0, pos));
  }
  if (need_mask_loop) {
    int pos = FindLastProducerIndex(functionProto, loop_deps) + 1;
    InsertNodeAt(functionProto, mask_loop_node, std::max(0, pos));
  }
  if (need_prob_loop) {
    std::vector<std::string> prob_deps = {"ProbFlat", "N", "CondInit", "B", "H", "L", "S"};
    int pos = FindLastProducerIndex(functionProto, prob_deps) + 1;
    InsertNodeAt(functionProto, prob_loop_node, std::max(0, pos));
  }

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
            "Defaults to float32 for float16/bfloat16 inputs, otherwise uses input type. "
            "Must be explicitly specified for non-float types.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "enable_gqa",
            "Enable Grouped Query Attention. "
            "0 (default): requires Hq == Hkv. "
            "1: K/V heads are broadcast to query heads (Hq must be divisible by Hkv).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "mask_value",
            "Value for masked scores before softmax. Defaults to -infinity.",
            AttributeProto::FLOAT,
            -std::numeric_limits<float>::infinity())
        .Attr(
            "score_mod",
            "Optional score modifier subgraph with 5 scalar inputs: "
            "(score, batch, head, q_idx, k_idx) -> score_out. "
            "score uses softmax_precision type; indices are INT64.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "mask_mod",
            "Optional mask modifier subgraph with 4 scalar inputs: "
            "(batch, head, q_idx, k_idx) -> mask_out (BOOL). "
            "All inputs are INT64 scalars.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "prob_mod",
            "Optional probability modifier subgraph with 5 scalar inputs: "
            "(prob, batch, head, q_idx, k_idx) -> prob_out. "
            "prob uses softmax_precision type; indices are INT64.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q, K, V to float tensors.")
        .TypeAndShapeInferenceFunction(FlexAttentionShapeInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(BuildFlexAttentionFunctionBody));

} // namespace ONNX_NAMESPACE
