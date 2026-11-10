```toml
name = "CausalConvWithState"
domain = "ai.onnx"

[attributes.activation]
type = "STRING"
desc = "Fused activation function. One of: 'silu', 'swish', 'none'. Default is 'none'."
default = "none"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor with shape (batch_size, channels, length). Channels-first layout."
differentiation = "differentiable"

[[inputs]]
name = "weight"
type = "T"
desc = "Depthwise convolution kernel with shape (channels, 1, k) where k is the kernel size. The middle dim of size 1 follows the ONNX `Conv` weight layout `(M, C/group, k1, ..., kn)`: since this op is always depthwise, `group = channels`, so `C/group = 1`. Keeping this layout makes the weight tensor a drop-in for a depthwise `Conv(group=channels)` weight, so `Conv` <-> `CausalConvWithState` rewrites require no reshape."
differentiation = "differentiable"

[[inputs]]
name = "bias"
type = "T"
desc = "Optional per-channel bias with shape (channels)."
option = "optional"
differentiation = "differentiable"

[[inputs]]
name = "past_state"
type = "T"
desc = "Carry state from previous step with shape (batch_size, channels, k - 1). If not provided, padding is zero."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Convolution output with same shape as input."
differentiation = "differentiable"

[[outputs]]
name = "present_state"
type = "T"
desc = "Updated carry state with shape (batch_size, channels, k - 1). Contains the last (k - 1) values of the effective padded/concatenated sequence along the causal axis, including any values from past_state or zero-padding when the current input is shorter than k - 1."
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "Constrain input and output types to float tensors."
```

Stateful causal 1D depthwise convolution.

Used by Gated DeltaNet (Qwen3.5) and Mamba (Jamba, FalconMamba) as a preprocessing step.
Replaces the 3-op pattern (Concat + Conv + Slice) with a single fused operation.

The convolution is causal (looks only at current and past positions) and depthwise
(each channel is convolved independently with its own kernel).

The input, weight, past_state, output, and present_state tensors are rank-3 with
shape (batch_size, channels, length). The optional bias input is rank-1 with
shape (channels). For higher-dimensional data, use Reshape nodes before and
after this operator to pack extra dimensions into the batch or channel axis.

Weight layout: (channels, 1, k) for depthwise convolution.
The carry state stores the last (k-1) positions for incremental decode.

The optional activation attribute supports fused SiLU/Swish activation.
