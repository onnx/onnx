```toml
name = "LinearAttention"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::LinearAttentionShapeInference_opset27"
function_body_builders = { 27 = "defs::nn::utils::LinearAttentionFunctionBody_opset27" }

[attributes.chunk_size]
type = "INT"
desc = "Chunk size for the chunk-parallel WY decomposition during prefill (T>1). Tuning hint; does not affect output correctness."
default = 64

[attributes.kv_num_heads]
type = "INT"
desc = "Number of key/value heads. Always required."
required = true

[attributes.q_num_heads]
type = "INT"
desc = "Number of query heads. Always required."
required = true

[attributes.scale]
type = "FLOAT"
desc = "Output scaling factor. When 0.0 (default), derives d_k = query.shape[-1] / q_num_heads and uses 1/sqrt(d_k). Set explicitly to override."
default = 0.0

[attributes.update_rule]
type = "STRING"
desc = "The update rule for the linear attention recurrence. One of: 'linear', 'gated', 'delta', 'gated_delta'. Default is 'gated_delta'."
default = "gated_delta"

[[inputs]]
name = "query"
type = "T"
desc = "Query vectors with 3D packed shape (B, T, H_q * d_k). Heads are packed into the last dimension."
differentiation = "differentiable"

[[inputs]]
name = "key"
type = "T"
desc = "Key vectors with 3D packed shape (B, T, H_kv * d_k). Should be L2-normalized for delta/gated_delta modes."
differentiation = "differentiable"

[[inputs]]
name = "value"
type = "T"
desc = "Value vectors with 3D packed shape (B, T, H_kv * d_v)."
differentiation = "differentiable"

[[inputs]]
name = "past_state"
type = "S"
desc = "Recurrent state from previous step with shape (B, H_kv, d_k, d_v). Always 4D. If not provided, defaults to zeros."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "decay"
type = "T"
desc = "Exponential decay gate in log-space. 3D packed shape: (B, T, H_kv * d_k) for per-key-dimension decay (GLA/RWKV-6), or (B, T, H_kv) for per-head scalar decay (DeltaNet/RetNet). Required for 'gated' and 'gated_delta' modes."
option = "optional"
differentiation = "differentiable"

[[inputs]]
name = "beta"
type = "T"
desc = "Update rate (sigmoid output). 3D packed shape: (B, T, H_kv) or (B, T, 1). Required for 'delta' and 'gated_delta' modes."
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Attention output with 3D packed shape (B, T, H_q * d_v)."
differentiation = "differentiable"

[[outputs]]
name = "present_state"
type = "S"
desc = "Updated recurrent state with shape (B, H_kv, d_k, d_v). Always 4D."
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
desc = "Constrain activation input and output types to float16, bfloat16, or float32 tensors."

[type_constraints.S]
types = ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
desc = "Constrain state types to float16, bfloat16, or float32 tensors. Should be float32 or the same as T for numerical stability on long sequences."
```

Unified linear attention operator for autoregressive decoding (T=1) and prefill (T>1).

The query, key, value, and (where applicable) decay/beta inputs use 3D packed format
[B, T, H*D], where heads are flattened into the last dimension; q_num_heads and
kv_num_heads are always required and are used to unpack to 4D internally for computation.
The optional past_state and present_state are 4D with shape (B, H_kv, d_k, d_v).

Group-query attention (GQA) is supported: q_num_heads must be a positive multiple of
kv_num_heads. When q_num_heads == kv_num_heads this reduces to multi-headed linear
attention; when q_num_heads > kv_num_heads each KV head (and its recurrent state) is
shared by `q_num_heads / kv_num_heads` query heads (multi-query attention is the
special case kv_num_heads == 1).

The update_rule attribute selects the recurrence type:
- "linear": S_t = S_{t-1} + k_t ⊗ v_t; o_t = scale * q_t^T S_t
- "gated": S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t; o_t = scale * q_t^T S_t
- "delta": S_t = S_{t-1} + β_t * k_t ⊗ (v_t - S_{t-1}^T k_t); o_t = scale * q_t^T S_t
- "gated_delta": S_t = exp(g_t) * S_{t-1} + β_t * k_t ⊗ (v_t - exp(g_t) * S_{t-1}^T k_t); o_t = scale * q_t^T S_t

where g_t is the decay (in log-space), β_t is the update rate, and ⊗ denotes outer product.

Semantics: Equivalent to running the recurrent update sequentially for each token,
but may be implemented using chunk-parallel algorithms for GPU efficiency.
