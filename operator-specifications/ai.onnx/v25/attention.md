```toml
name = "Attention"
domain = "ai.onnx"

[attributes.is_causal]
type = "INT"
desc = "If set to `1`, causal masking is applied. For a square Q/K (no cache offset) this is a lower-triangular matrix. In general the mask is bottom-right (offset-aware): query in-block index `i` attends key `j` iff `j <= i + offset`, where `offset` is the count of valid keys preceding the query block (`past_sequence_length` for an internal `past_key` cache, or `nonpad_kv_seqlen - q_sequence_length` per batch for an external cache). When `offset = 0` this reduces to the lower-triangular (top-left) mask."
default = 0

[attributes.kv_num_heads]
type = "INT"
desc = "Number of heads of key and value. Must be used with 3D inputs of Q, K and V. "

[attributes.left_window_size]
type = "INT"
desc = "Maximum number of positions to the left of the current absolute query position that may be attended. A value of `0` allows the current position but no preceding position, while `-1` leaves the left side unbounded. This bound is composed with `is_causal` and `attn_mask`."
default = -1

[attributes.q_num_heads]
type = "INT"
desc = "Number of heads of query. Must be used with 3D inputs of Q, K and V. "

[attributes.qk_matmul_output_mode]
type = "INT"
desc = "Determines what the optional 4th output contains: `0` (default): raw QK matmul result; `1`: after softcap (before bias addition); `2`: QK + softcap + bias; `3`: post-softmax probabilities (after fully-masked-row guard). In mode `3`, a fully-masked query row (every key disallowed) is a zero row, consistent with the corresponding row of the primary output `Y`. The mode-`3` output is emitted at the operator's output precision (`T1`); when `softmax_precision` differs from `T1` this is a cast of the softmax result to `T1`."
default = 0

[attributes.right_window_size]
type = "INT"
desc = "Maximum number of positions to the right of the current absolute query position that may be attended. A value of `0` allows the current position but no following position, while `-1` leaves the right side unbounded. Set `is_causal=0` to use a positive right window."
default = -1

[attributes.scale]
type = "FLOAT"
desc = "Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent [numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul."

[attributes.softcap]
type = "FLOAT"
desc = "Soft cap for attention logits, applied as `softcap * tanh(logits / softcap)`. Default value of `0.0` means no soft capping is applied. The soft cap is applied before mask / bias addition and softmax."
default = 0.0

[attributes.softmax_precision]
type = "INT"
desc = "Specifies the precision for softmax computation. If provided, the attention weights will be cast to this type before softmax and then cast back to the original type. Supported values are: `1` (FLOAT), `10` (FLOAT16), `11` (DOUBLE), `16` (BFLOAT16)."

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
desc = "Attention mask. Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` where `total_sequence_length = past_sequence_length + kv_sequence_length`. The last dimension can also be shorter than `total_sequence_length` and will be padded to `total_sequence_length` with negative infinity. Two types of masks are supported: a boolean mask where a value of `True` indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score."
option = "optional"

[[inputs]]
name = "past_key"
type = "T1"
desc = "Past state for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`. Must be used together with `past_value` input."
option = "optional"

[[inputs]]
name = "past_value"
type = "T2"
desc = "Past state for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`. Must be used together with `past_key` input."
option = "optional"

[[inputs]]
name = "nonpad_kv_seqlen"
type = "tensor(int64)"
desc = "A vector of integers of shape `(batch_size,)` that indicates the number of valid (i.e., non-padding) tokens in each sample. A padding mask can be derived from this. This should not be used together with `past_key` and `past_value` inputs or `present_key` and `present_value` outputs (see the KV cache use cases in the operator description)."
option = "optional"

[[outputs]]
name = "Y"
type = "T1"
desc = "The output tensor. 4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`"

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

`left_window_size` and `right_window_size` independently restrict the keys visible to each query. A query at absolute position `p = offset + query_index` attends keys `j` satisfying `p - left_window_size <= j <= p + right_window_size` for each nonnegative bound. A value of `-1` leaves that side unbounded. For example, `(left_window_size=2, right_window_size=0)` is a causal left-looking window containing the current key and two preceding keys, while `(left_window_size=2, right_window_size=1)` is an asymmetric bidirectional window. Window bounds are composed with `is_causal` and `attn_mask`; when `is_causal=1`, the causal upper bound still excludes future keys.

```
  2D sliding-window mask for Attention (opset 25)
   S_q=4 queries, S_k=6 keys, left_window_size=2, right_window_size=1, offset=0

          k0  k1  k2  k3  k4  k5
         +----+----+----+----+----+----+
    q0   | ## | ## |    |    |    |    |
         +----+----+----+----+----+----+
    q1   | ## | ## | ## |    |    |    |
         +----+----+----+----+----+----+
    q2   | ## | ## | ## | ## |    |    |
         +----+----+----+----+----+----+
    q3   |    | ## | ## | ## | ## |    |
         +----+----+----+----+----+----+

   q0 attends {k0,k1}, q1 attends {k0,k1,k2}, q2 attends {k0,k1,k2,k3},
   q3 attends {k1,k2,k3,k4}.
```

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
