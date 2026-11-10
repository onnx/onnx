```toml
name = "TensorScatter"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Sequence dimension of the `past_cache` and `update` tensors. It cannot be 0 (the batch dimension). Default is -2."
default = -2

[attributes.mode]
type = "STRING"
desc = "Write mode of cache update. Supported modes include `linear` and `circular`. `linear` mode requires write_indices+sequence_length<=max_sequence_length. For `circular` mode, the updates happen in wrap-around fashion, ie, the update index is modulo `max_sequence_length`"
default = "linear"

[[inputs]]
name = "past_cache"
type = "T"
desc = "Past state cache for key or value with shape `(batch_size, D1, D2, ..., max_sequence_length, ..., Dn)`."
differentiation = "differentiable"

[[inputs]]
name = "update"
type = "T"
desc = "New update tensor with shape `(batch_size, D1, D2, ..., sequence_length, ..., Dn)`."
differentiation = "differentiable"

[[inputs]]
name = "write_indices"
type = "tensor(int64)"
desc = "Write indices for the incoming update tensor in the cache. Shape is `(batch_size,)`. Assumed to be all zeros if not provided."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "present_cache"
type = "T"
desc = "Updated cache. Same shape as `past_cache`."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir12"
desc = "Constrain input and output types to any tensor type."
```

TensorScatter is a generic tensor update operation, motivated by the requirements for KV cache updates for Attention
ops commonly found in LLMs. It is a functional operation that models an in-place update to a KV cache buffer.

The past and present cache tensors have the same shape (batch_size, D1, D2, ..., max_sequence_length, ..., Dn), with
the sequence dimension (indicated by the `axis` attribute) being max_sequence_length, so the sizes of these tensors do
not need to grow between iterations. The `update` tensor's shape only differs from the cache tensors in the sequence
dimension: (batch_size, D1, D2, ..., sequence_length, ..., Dn), where sequence_length <= max_sequence_length.

The optional `write_indices` input indicates the write index for each sample in the batch, assumed to be zero
if not provided. When the `mode` attribute is set to "circular", the write index is modulo max_sequence_length.
The operation can be described using the following pseudocode:

```
for prefix_idx in np.ndindex(past_cache.shape[:axis]):
    batch_idx = prefix_idx[0]
    for sequence_idx in range(sequence_length):
        cache_sequence_idx = write_indices[batch_idx] + sequence_idx
        if mode == "circular":
            cache_sequence_idx = cache_sequence_idx % max_sequence_length
        cache_idx = (*prefix_idx, cache_sequence_idx)
        update_idx = (*prefix_idx, sequence_idx)
        present_cache[cache_idx] = update[update_idx]
```

During the prefill phase of attention, only the first two inputs are needed. During the decode phase, `write_indices`
is also needed so that the incoming key or value update can be appended after the last valid token for each sample
in the batch.
