```toml
name = "FlexAttention"
domain = "ai.onnx.preview"
support_level = "experimental"
node_determinism = "deterministic"

[attributes.prob_mod]
type = "GRAPH"
desc = "Optional probability modifier subgraph with 1 rank-4 tensor input and 1 rank-4 tensor output of the same shape and element type: (probs) -> probs_out. probs has softmax_precision element type and shape (B, Hq, L, S). The output must preserve the input shape."

[attributes.scale]
type = "FLOAT"
desc = "Scaling factor for Q*K^T. Defaults to 1/sqrt(head_size)."

[attributes.score_mod]
type = "GRAPH"
desc = "Optional score modifier subgraph with 1 rank-4 tensor input and 1 rank-4 tensor output of the same shape and element type: (scores) -> scores_out. scores has softmax_precision element type and shape (B, Hq, L, S). The output must preserve the input shape."

[attributes.softmax_precision]
type = "INT"
desc = "Floating-point precision for softmax computation. Defaults to float32 for non-double inputs, otherwise uses double. Must be explicitly specified for non-float types."

[[inputs]]
name = "Q"
type = "T1"
desc = "Query tensor with shape `(batch_size, q_num_heads, q_seq_len, head_size)`."

[[inputs]]
name = "K"
type = "T1"
desc = "Key tensor with shape `(batch_size, kv_num_heads, kv_seq_len, head_size)`."

[[inputs]]
name = "V"
type = "T1"
desc = "Value tensor with shape `(batch_size, kv_num_heads, kv_seq_len, v_head_size)`."

[[outputs]]
name = "Y"
type = "T1"
desc = "Output tensor with shape `(batch_size, q_num_heads, q_seq_len, v_head_size)`."

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain Q, K, V to float tensors."
```

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
