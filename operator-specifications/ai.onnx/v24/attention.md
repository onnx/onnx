```toml
name = "Attention"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput"
function_body_builders = { 24 = "defs::nn::utils::AttentionFunctionBody_opset24" }

[attributes.is_causal]
type = "INT"
desc = "If set to `1`, causal masking is applied. For a square Q/K (no cache offset) this is a lower-triangular matrix. In general the mask is bottom-right (offset-aware): query in-block index `i` attends key `j` iff `j <= i + offset`, where `offset` is the count of valid keys preceding the query block (`past_sequence_length` for an internal `past_key` cache, or `nonpad_kv_seqlen - q_sequence_length` per batch for an external cache). When `offset = 0` this reduces to the lower-triangular (top-left) mask."
default = 0

[attributes.kv_num_heads]
type = "INT"
desc = "Number of heads of key and value. Must be used with 3D inputs of Q, K and V. "

[attributes.q_num_heads]
type = "INT"
desc = "Number of heads of query. Must be used with 3D inputs of Q, K and V. "

[attributes.qk_matmul_output_mode]
type = "INT"
desc = "If set to `0`, qk_matmul_output is the output of qk matmul. If set to `1`, qk_matmul_output is the output after the softcap operation (before mask addition). If set to `2`, qk_matmul_output includes the attention mask and softcap (if provided) applied to the output of qk matmul. If set to `3`, qk_matmul_output is the output after the softmax operation. In mode `3`, a fully-masked query row (every key disallowed) is a zero row, consistent with the corresponding row of the primary output `Y`: the fully-masked-row guard is applied before this output is produced. The mode-`3` output is emitted at the operator's output precision (`T1`); when `softmax_precision` differs from `T1` this is a cast of the softmax result to `T1`. Default value is 0."
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
desc = "Attention mask. Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length.` The last dimension can also be shorter than `total_sequence_length` and will be padded to `total_sequence_length` with negative infinity. Two types of masks are supported: a boolean mask where a value of `True` indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score."
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

[[inputs]]
name = "nonpad_kv_seqlen"
type = "tensor(int64)"
desc = "A vector of integers of shape `(batch_size,)` that indicates the number of valid (ie, non-padding) tokens in each sample. A padding mask can be derived from this. This should not be used together with `past_key` and `past_value` inputs or `present_key` and `present_value` outputs (See the KV cache use cases in the operator description)."
option = "optional"
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
2) If `is_causal` is set to `1`, causal masking is applied with bottom-right (offset-aware) alignment: query `i` attends key `j` iff `j <= i + offset`, as illustrated below.

```
  2D causal mask for Attention (PR onnx/onnx#8068)
   S_q=4 queries, S_k=8 keys
   Rule: query i attends key j iff j <= i + offset
         offset = nonpad_kv_seqlen - S_q

   nonpad_kv_seqlen=4, offset=4-4=0

          k0  k1  k2  k3  k4  k5  k6  k7
         +----+----+----+----+----+----+----+----+
    q0   | ## |    |    |    |    |    |    |    |
         +----+----+----+----+----+----+----+----+
    q1   | ## | ## |    |    |    |    |    |    |
         +----+----+----+----+----+----+----+----+
    q2   | ## | ## | ## |    |    |    |    |    |
         +----+----+----+----+----+----+----+----+
    q3   | ## | ## | ## | ## |    |    |    |    |
         +----+----+----+----+----+----+----+----+


   nonpad_kv_seqlen=8, offset=8-4=4

          k0  k1  k2  k3  k4  k5  k6  k7
         +----+----+----+----+----+----+----+----+
    q0   | ## | ## | ## | ## | ## |    |    |    |
         +----+----+----+----+----+----+----+----+
    q1   | ## | ## | ## | ## | ## | ## |    |    |
         +----+----+----+----+----+----+----+----+
    q2   | ## | ## | ## | ## | ## | ## | ## |    |
         +----+----+----+----+----+----+----+----+
    q3   | ## | ## | ## | ## | ## | ## | ## | ## |
         +----+----+----+----+----+----+----+----+
```

With `nonpad_kv_seqlen=4` (offset=0), the mask is the standard lower-triangular. With `nonpad_kv_seqlen=8` (offset=4), the diagonal shifts right by 4, so each query sees the 4 additional valid cached keys.

`offset` is the count of valid keys preceding the current query block: `offset = past_sequence_length` when `past_key` is provided; `offset = nonpad_kv_seqlen - q_sequence_length` (per batch) when an external cache is indicated by `nonpad_kv_seqlen` without `past_key`; `offset = 0` when neither is provided (the no-cache case, which reduces to the standard lower-triangular mask). When `offset < 0` (`nonpad_kv_seqlen < q_sequence_length`, i.e. more query tokens than cached keys) the leading query rows have an empty key set (no key satisfies `j <= i + offset`) and are fully masked. The causal frontier is computed independently of `attn_mask` and is then composed with it additively: a boolean `attn_mask` intersects the allowed set (its disallowed positions contribute `-inf` to the bias), while a float `attn_mask` is added to the attention scores rather than disabling positions. A fully-masked query row (no key attended, including the negative-offset leading rows) produces a zero output row, not `NaN`, for both `Y` and the mode-`3` `qk_matmul_output` debug output; the mode-`3` `qk_matmul_output` is emitted at the operator's output precision (`T1`).

Errata (in-place behavioral correction, no opset bump): the reference implementation and backend tests were incorrect when `nonpad_kv_seqlen != q_sequence_length` (nonzero bottom-right offset, top-left instead of bottom-right causal alignment) and produced `NaN` for fully-masked rows; corrected in version 1.23. This fixed three behaviors described above: external-cache bottom-right causal alignment (`offset = nonpad_kv_seqlen - q_sequence_length`), zero (non-`NaN`) output for fully-masked rows including the mode-`3` `qk_matmul_output`, and the mode-`3` `qk_matmul_output` precision (`T1`).

With respect to KV cache update, this operator allows the following two use cases:

1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
key and value, respectively. Note that this only works correctly for the special case where the past key and value
do not contain padded tokens.
2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
as the maximum value of `nonpad_kv_seqlen`.

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
