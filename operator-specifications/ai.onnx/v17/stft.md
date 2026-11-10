```toml
name = "STFT"
domain = "ai.onnx"
shape_inference = "defs::math::utils::STFTShapeInference_opset17"

[attributes.onesided]
type = "INT"
desc = "If onesided is 1, only values for w in [0, 1, 2, ..., floor(n_fft/2) + 1] are returned because the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., X[m, w] = X[m, n_fft-w]*. Note if the input or window tensors are complex, then onesided output is not possible. Enabling onesided with real inputs performs a Real-valued fast Fourier transform (RFFT). When invoked with real or complex valued input, the default value is 1. Values can be 0 or 1."
default = 1

[[inputs]]
name = "signal"
type = "T1"
desc = "Input tensor representing a real or complex valued signal. For real input, the following shape is expected: [batch_size][signal_length][1]. For complex input, the following shape is expected: [batch_size][signal_length][2], where [batch_size][signal_length][0] represents the real component and [batch_size][signal_length][1] represents the imaginary component of the signal. The tensor is expected to have rank 3."
differentiation = "non_differentiable"

[[inputs]]
name = "frame_step"
type = "T2"
desc = "A scalar representing the number of samples to step between successive DFTs."
differentiation = "non_differentiable"

[[inputs]]
name = "window"
type = "T1"
desc = "An optional 1-D tensor representing the window function to be applied to each frame of the signal before computing the DFT. The length of the window (window.shape[0]) determines the frame length when `frame_length` is not specified. If both `window` and `frame_length` are provided, the length of the `window` must equal `frame_length`. When omitted, a rectangular (all-ones) window of length `frame_length` is used."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "frame_length"
type = "T2"
desc = "An optional scalar representing the length of each frame (i.e., the DFT size). When omitted and `window` is provided, `frame_length` is inferred from `window.shape[0]`. When both `window` and `frame_length` are omitted, `frame_length` defaults to `signal_length`. If both `frame_length` and `window` are provided, the length of the `window` must equal `frame_length`."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T1"
desc = "The Short-time Fourier Transform of the signal. The number of frames in the output is `frames = floor((signal_length - frame_length) / frame_step) + 1`. If onesided is 1, the output has the shape: [batch_size][frames][dft_unique_bins][2], where dft_unique_bins is frame_length // 2 + 1 (the unique components of the DFT). If onesided is 0, the output has the shape: [batch_size][frames][frame_length][2], where frame_length is the length of the DFT. The last dimension of size 2 represents the real and imaginary parts of each complex value."
differentiation = "non_differentiable"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain signal and output to float tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain scalar length types to int64_t."
```

Computes the Short-time Fourier Transform of the signal.

The STFT is computed by sliding a window of length `frame_length` over the signal with a
step size of `frame_step`, computing a DFT of each windowed frame.

The number of frames in the output is computed as:

  `frames = floor((signal_length - frame_length) / frame_step) + 1`

Constraints on inputs:
- `frame_step` must be a scalar.
- `frame_length` must be a scalar. When omitted and `window` is provided, `frame_length`
  is inferred from `window.shape[0]`. When both `window` and `frame_length` are omitted,
  `frame_length` defaults to `signal_length`.
- `window` must be a 1-D tensor. When omitted, a rectangular (all-ones) window of length
  `frame_length` is used. When both `window` and `frame_length` are provided, the length
  of the `window` tensor must equal `frame_length`.
