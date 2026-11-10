```toml
name = "RotaryEmbedding"
domain = "ai.onnx"
node_determinism = "deterministic"

[attributes.interleaved]
type = "INT"
desc = "Rotate using interleaved pattern. Default value is 0 (False)."
default = 0

[attributes.num_heads]
type = "INT"
desc = "Number of attention heads. Must be provided when input is a 3D tensor. "

[attributes.rotary_embedding_dim]
type = "INT"
desc = "Rotary embedding dimension used to apply partial rotary embeddings."
default = 0

[[inputs]]
name = "X"
type = "T"
desc = "The input tensor representing the token embeddings. 4D tensor with shape `(batch_size, num_heads, sequence_length, head_size)` or 3D tensor with shape `(batch_size, sequence_length, hidden_size)`. For cases with a 4D input tensor, `head_size` has to be even. For cases with a 3D input tensor, `num_heads` attribute must be provided and `hidden_size` must be an even multiple of `num_heads` where `hidden_size = num_heads * head_size`"

[[inputs]]
name = "cos_cache"
type = "T"
desc = "The cosine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are provided. 3D tensor with shape `(batch_size, sequence_length, head_size / 2)` for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are not provided. `max_position_id_plus_1` is a parameter to the model."

[[inputs]]
name = "sin_cache"
type = "T"
desc = "The sine values for the rotation. 2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are provided. 3D tensor with shape `(batch_size, sequence_length, head_size / 2)` for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are not provided. `max_position_id_plus_1` is a parameter to the model."

[[inputs]]
name = "position_ids"
type = "M"
desc = "The position indices for the tokens. 2D tensor with shape `(batch_size, sequence_length)`"
option = "optional"

[[outputs]]
name = "Y"
type = "T"
desc = "Tensor with same shape as input."

[type_constraints.T]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "Constrain input and output types to float tensors."

[type_constraints.M]
types = ["tensor(int64)"]
desc = "Constrain input and output types to integer tensors."
```

RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.org/pdf/2104.09864.
The key advantage of RoPE is that it allows the model to understand both the absolute position of a token and the relative distances
between tokens. This is achieved through a rotational mechanism where the extent of rotation is computed based on the token's absolute position (position_ids).

The rotational mechanism is defined by sine and cosine functions that are used to represent the rotation angles.
For each token in the sequence, its positional embedding is computed by rotating its embedding vector. This is done by splitting the
embedding vector either into two halves or interleaving every alternate token and applying the rotation matrix to each half of the embedding vector.
The rotation matrix is parameterized by the token's position in the sequence. The rotated halves of the embedding vector are concatenated
to form the final positional embedding for each token. The rotated positional embeddings are used in the self-attention mechanism.
The rotation ensures that the model captures both absolute and relative positional information.

Rotary embeddings are defined using the following algorithm:

```python
def rotary_embedding(
    input: np.ndarray,
    cos_cache: np.ndarray,
    sin_cache: np.ndarray,
    position_ids: np.ndarray | None = None,
    interleaved=None,
    rotary_embedding_dim=None,
    num_heads=None,
) -> np.ndarray:
    original_input_shape = input.shape
    # First ensure input to be processed has shape [batch_size, seq_len, num_heads, head_size]
    if len(input.shape) == 4:
        input = np.transpose(input, (0, 2, 1, 3))
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    if len(input.shape) == 3:
        hidden_size = input.shape[2]
        assert num_heads != 0
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input = np.reshape(input, new_shape)
    assert len(input.shape) == 4
    head_size = input.shape[3]

    # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    if rotary_embedding_dim is None or rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos_cache = cos_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
        sin_cache = sin_cache[
            position_ids
        ]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]

    # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    if cos_cache.shape[-1] != rotary_embedding_dim_half:
        raise ValueError(
            f"Last dimension of cos cache ({cos_cache.shape[-1]}) does not match rotary_embedding_dim/2 ({rotary_embedding_dim_half})."
        )
    if sin_cache.shape[-1] != rotary_embedding_dim_half:
        raise ValueError(
            f"Last dimension of sin cache ({sin_cache.shape[-1]}) does not match rotary_embedding_dim/2 ({rotary_embedding_dim_half})."
        )

    cos_cache = np.expand_dims(
        cos_cache, axis=2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin_cache = np.expand_dims(
        sin_cache, axis=2
    )  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the input in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)

    # Calculate real and imaginary values
    real = (cos_cache * x1) - (sin_cache * x2)
    imag = (sin_cache * x1) + (cos_cache * x2)

    # Inserted rotated embeddings back to the original input
    if interleaved:
        # x_rotate[:, :, :, 0::2] = real
        # x_rotate[:, :, :, 1::2] = imag
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        x_rotate_concat = np.concatenate((real, imag), axis=-1)
        x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)
    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if len(original_input_shape) == 3:
        output = np.reshape(output, original_input_shape)
    else:
        output = np.transpose(output, (0, 2, 1, 3))
    return output
```
