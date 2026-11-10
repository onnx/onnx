```toml
name = "Attention"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput"
function_body_builders = { 23 = "defs::nn::utils::AttentionFunctionBody_opset23" }

[attributes.is_causal]
type = "INT"
desc = "If set to `1`, causal masking is applied with bottom-right (offset-aware) alignment: query `i` attends key `j` iff `j <= i + past_sequence_length` (the count of cached keys in `past_key`); for a square Q/K this is the standard lower-triangular mask."
default = 0

[attributes.kv_num_heads]
type = "INT"
desc = "Number of heads of key and value. Must be used with 3D inputs of Q, K and V. "

[attributes.q_num_heads]
type = "INT"
desc = "Number of heads of query. Must be used with 3D inputs of Q, K and V. "

[attributes.qk_matmul_output_mode]
type = "INT"
desc = "If set to `0`, qk_matmul_output is the output of qk matmul. If set to `1`, qk_matmul_output is the output after the softcap operation (before mask addition). If set to `2`, qk_matmul_output includes the attention mask and softcap (if provided) applied to the output of qk matmul. If set to `3`, qk_matmul_output is the output after the softmax operation. In mode `3`, a fully-masked query row (every key disallowed, e.g. an all-`False` boolean `attn_mask` row) is a zero row, consistent with the corresponding row of the primary output `Y`: the fully-masked-row guard is applied before this output is produced. The mode-`3` output is emitted at the operator's output precision (`T1`); when `softmax_precision` differs from `T1` this is a cast of the softmax result to `T1`. Default value is 0."
default = 0

[attributes.scale]
type = "FLOAT"
desc = "Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent [numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul."

[attributes.softcap]
type = "FLOAT"
desc = "Softcap value for attention weights. Default value is 0."
default = 0.0

[attributes.softmax_precision]
type = "INT"
desc = "The floating-point precision used in softmax computation. If softmax precision is not provided, the same precision as the input of softmax (Q and K) is used."

[[inputs]]
name = "Q"
type = "T1"
desc = "Query tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, q_hidden_size)`. For cases with a 3D input tensor, `q_hidden_size = q_num_heads * head_size`"

[[inputs]]
name = "K"
type = "T1"
desc = "Key tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, k_hidden_size)`. For cases with a 3D input tensor, `k_hidden_size = kv_num_heads * head_size`"

[[inputs]]
name = "V"
type = "T2"
desc = "Value tensor. 4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, v_hidden_size)`. For cases with a 3D input tensor, `v_hidden_size = kv_num_heads * v_head_size`"

[[inputs]]
name = "attn_mask"
type = "U"
desc = "Attention mask. Shape must be broadcastable to 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length.` Two types of masks are supported. A boolean mask where a value of `True` indicates that the element should take part in attention. Also supports a float mask of the same type as query, key, value that is added to the attention score."
option = "optional"

[[inputs]]
name = "past_key"
type = "T1"
desc = "past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`"
option = "optional"

[[inputs]]
name = "past_value"
type = "T2"
desc = "past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`"
option = "optional"

[[outputs]]
name = "Y"
type = "T1"
desc = "The output tensor . 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`"

[[outputs]]
name = "present_key"
type = "T1"
desc = "Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` where `total_sequence_length = past_sequence_length + kv_sequence_length`."
option = "optional"

[[outputs]]
name = "present_value"
type = "T2"
desc = "Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` where `total_sequence_length = past_sequence_length + kv_sequence_length`."
option = "optional"

[[outputs]]
name = "qk_matmul_output"
type = "T1"
desc = "The output of QK matmul. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length`."
option = "optional"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain Q and K inputs types to float tensors."

[type_constraints.T2]
types = "all_float_types_ir4"
desc = "Constrain V input types to float tensors."

[type_constraints.U]
types = "all_non_complex_numeric_types_plus_bool_ir4"
desc = "Constrain output 'mask' types to boolean tensors and input types."
```

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
2) If `is_causal` is set to `1`, causal masking is applied with bottom-right (offset-aware) alignment: query `i` attends key `j` iff `j <= i + past_sequence_length` (the count of cached keys in `past_key`); for a square Q/K this is the standard lower-triangular mask. The causal frontier is computed independently of the `attn_mask` input and is then composed with it additively, by summing their attention biases: a boolean `attn_mask` intersects the allowed set (its disallowed positions contribute `-inf` to the bias), while a float `attn_mask` is added to the attention scores rather than disabling positions. A fully-masked query row (every key's combined additive bias is `-inf`, e.g. an all-`False` boolean `attn_mask` row) produces a zero output row (matching prevailing runtime practice), not `NaN`.

Errata (in-place behavioral correction, no opset bump): a fully-masked query row (e.g. an all-`False` boolean `attn_mask` row) now produces a zero output row instead of `NaN`, and the same zero-row guard applies to the mode-`3` `qk_matmul_output` debug output; this only replaces previously-`NaN` outputs. The mode-`3` `qk_matmul_output` is also now emitted at the operator's output precision (`T1`), matching the reference implementation, which affects only its dtype and only when `softmax_precision` differs from `T1`. No numerically useful, well-defined result of the released opset is otherwise changed.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
  softcap (if provided)     |
            |               |
 at_mask---Add              |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```
