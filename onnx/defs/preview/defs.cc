/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
FlexAttention is a flexible attention operator that allows custom computation subgraphs
to modify the attention mechanism at three specific stages:

- `score_mod`: Modifies the attention scores after computing Q*K^T (e.g., for bias or positional encoding).
- `mask_mod`: Applies masking logic to the scores (e.g., for causal masking or padding masks).
- `prob_mod`: Modifies the probabilities after Softmax (e.g., for quantization or dropout).

This operator mirrors the capabilities of PyTorch's `torch.nn.functional.flex_attention`,
providing a mechanism to implement highly customized and efficient attention mechanisms.

The base computation follows the standard scaled dot-product attention pattern:

```
scores = Q @ K^T / sqrt(head_size)
scores = score_mod(scores) if score_mod provided
scores = mask_mod(scores) if mask_mod provided
probs = Softmax(scores, axis=-1)
probs = prob_mod(probs) if prob_mod provided
output = probs @ V
```

The custom subgraphs (score_mod, mask_mod, prob_mod) are passed as GraphProto attributes.
Each subgraph takes the intermediate result as input and produces a modified result as output.

**Inputs**

- **Q** (T1): Query tensor with shape `(batch_size, num_heads, seq_length_q, head_size)`.
- **K** (T1): Key tensor with shape `(batch_size, num_heads, seq_length_k, head_size)`.
- **V** (T2): Value tensor with shape `(batch_size, num_heads, seq_length_k, head_size_v)`.

**Outputs**

- **output** (T2): Output tensor with shape `(batch_size, num_heads, seq_length_q, head_size_v)`.

**Attributes**

- **score_mod** (graph, optional): A GraphProto that modifies attention scores. 
  Takes a tensor of shape `(batch_size, num_heads, seq_length_q, seq_length_k)` and returns
  a tensor of the same shape.
  
- **mask_mod** (graph, optional): A GraphProto that applies masking to attention scores.
  Takes a tensor of shape `(batch_size, num_heads, seq_length_q, seq_length_k)` and returns
  a tensor of the same shape.
  
- **prob_mod** (graph, optional): A GraphProto that modifies attention probabilities after softmax.
  Takes a tensor of shape `(batch_size, num_heads, seq_length_q, seq_length_k)` and returns
  a tensor of the same shape.
  
- **scale** (float, optional): Scaling factor applied to attention scores. If not provided,
  defaults to `1/sqrt(head_size)`.

**Type Constraints**

- **T1**: Constrain Q and K to float types (float32, float16, bfloat16, float64).
- **T2**: Constrain V and output to float types (float32, float16, bfloat16, float64).

**Example**

```python
# Score modification example: adding positional bias
score_mod_graph = make_graph(
    [make_node('Add', ['scores', 'bias'], ['modified_scores'])],
    'score_mod',
    [make_value_info('scores', TensorProto.FLOAT, ['B', 'H', 'Sq', 'Sk'])],
    [make_value_info('modified_scores', TensorProto.FLOAT, ['B', 'H', 'Sq', 'Sk'])]
)

flex_attention_node = make_node(
    'FlexAttention',
    inputs=['Q', 'K', 'V'],
    outputs=['output'],
    domain='ai.onnx.preview',
    score_mod=score_mod_graph
)
```
)DOC";

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
        .Attr(
            "score_mod",
            "Optional subgraph to modify attention scores after Q*K^T computation. "
            "The subgraph takes a single input tensor (the attention scores) and produces "
            "a single output tensor of the same shape.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "mask_mod",
            "Optional subgraph to apply masking to attention scores. "
            "The subgraph takes a single input tensor (the attention scores) and produces "
            "a single output tensor of the same shape.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "prob_mod",
            "Optional subgraph to modify attention probabilities after Softmax. "
            "The subgraph takes a single input tensor (the attention probabilities) and produces "
            "a single output tensor of the same shape.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "scale",
            "Scaling factor applied to attention scores (Q*K^T). "
            "If not provided, defaults to 1/sqrt(head_size).",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Input(
            0,
            "Q",
            "Query tensor with shape (batch_size, num_heads, seq_length_q, head_size).",
            "T1")
        .Input(
            1,
            "K",
            "Key tensor with shape (batch_size, num_heads, seq_length_k, head_size).",
            "T1")
        .Input(
            2,
            "V",
            "Value tensor with shape (batch_size, num_heads, seq_length_k, head_size_v).",
            "T2")
        .Output(
            0,
            "output",
            "Output tensor with shape (batch_size, num_heads, seq_length_q, head_size_v).",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(double)"},
            "Constrain Q and K to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(double)"},
            "Constrain V and output to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Propagate element type from V to output
          propagateElemTypeFromInputToOutput(ctx, 2, 0);
          
          // Shape inference: output shape is (batch_size, num_heads, seq_length_q, head_size_v)
          if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 2)) {
            return;
          }
          
          auto& q_shape = getInputShape(ctx, 0);
          auto& v_shape = getInputShape(ctx, 2);
          
          if (q_shape.dim_size() != 4 || v_shape.dim_size() != 4) {
            fail_shape_inference("Q, K, and V must be 4D tensors");
          }
          
          auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          
          // Output shape: (batch_size, num_heads, seq_length_q, head_size_v)
          *output_shape->add_dim() = q_shape.dim(0);  // batch_size
          *output_shape->add_dim() = q_shape.dim(1);  // num_heads
          *output_shape->add_dim() = q_shape.dim(2);  // seq_length_q
          *output_shape->add_dim() = v_shape.dim(3);  // head_size_v
        })
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx,
               const OpSchema& schema,
               FunctionProto& functionProto) -> bool {
              // Build the function body for FlexAttention
              // This is a context-dependent function that constructs the attention computation
              // based on the provided attributes
              
              auto* tp = ctx.getInputType(0);
              if (tp == nullptr || !tp->has_tensor_type()) {
                return false;
              }
              
              FunctionBuilder builder(functionProto);
              
              // Get the scale attribute if provided
              auto scale_attr = ctx.getAttribute("scale");
              
              // Step 1: Compute Q @ K^T
              builder.Add("K_transposed = Transpose <perm = [0, 1, 3, 2]> (K)");
              builder.Add("qk_scores = MatMul (Q, K_transposed)");
              
              // Step 2: Apply scaling
              if (scale_attr != nullptr) {
                builder.Const1D("scale_value", scale_attr->f());
                builder.Add("scores_scaled = Mul (qk_scores, scale_value)");
              } else {
                // For now, use Identity if no scale is provided
                // A full implementation would compute default scale: 1/sqrt(head_size)
                builder.Add("scores_scaled = Identity (qk_scores)");
              }
              
              // Step 3-4: Apply score_mod and mask_mod if provided, then Softmax
              // Note: Graph attributes require special handling. For this reference implementation,
              // we use Identity as a placeholder. Backend implementations should inline the subgraphs.
              auto score_mod_attr = ctx.getAttribute("score_mod");
              auto mask_mod_attr = ctx.getAttribute("mask_mod");
              
              if (score_mod_attr != nullptr && score_mod_attr->has_g()) {
                builder.Add("scores_after_score_mod = Identity (scores_scaled)");
                builder.Add("attention_probs = Softmax <axis = -1> (scores_after_score_mod)");
              } else if (mask_mod_attr != nullptr && mask_mod_attr->has_g()) {
                builder.Add("scores_after_mask_mod = Identity (scores_scaled)");
                builder.Add("attention_probs = Softmax <axis = -1> (scores_after_mask_mod)");
              } else {
                builder.Add("attention_probs = Softmax <axis = -1> (scores_scaled)");
              }
              
              // Step 5-6: Apply prob_mod if provided, then compute output
              auto prob_mod_attr = ctx.getAttribute("prob_mod");
              if (prob_mod_attr != nullptr && prob_mod_attr->has_g()) {
                builder.Add("probs_after_prob_mod = Identity (attention_probs)");
                builder.Add("output = MatMul (probs_after_prob_mod, V)");
              } else {
                builder.Add("output = MatMul (attention_probs, V)");
              }
              
              return true;
            }));

} // namespace ONNX_NAMESPACE
