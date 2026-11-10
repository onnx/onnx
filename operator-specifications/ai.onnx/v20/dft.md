```toml
name = "DFT"
domain = "ai.onnx"

[attributes.inverse]
type = "INT"
desc = "Whether to perform the inverse discrete Fourier Transform. Default is 0, which corresponds to `false`."
default = 0

[attributes.onesided]
type = "INT"
desc = "If `onesided` is `1`, only values for `k` in `[0, 1, 2, ..., floor(n_fft/2) + 1]` are used or returned because the real-to-complex Fourier transform satisfies the conjugate symmetry, i.e., `X[m, k] = X[m, n_fft-k]*`, where `m` denotes \"all other dimensions\" DFT was not applied on. When `onesided=1` and `inverse=0` (forward DFT), only real input is supported and a one-sided complex spectrum is returned (RFFT). When `onesided=1` and `inverse=1` (inverse DFT), only complex input is supported and a full real signal is returned (IRFFT). Value can be `0` or `1`. Default is `0`."
default = 0

[[inputs]]
name = "input"
type = "T1"
desc = "For real input, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]`. For complex input, the following shape is expected: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]`. The final dimension represents the real and imaginary parts of the value in that order."
differentiation = "non_differentiable"

[[inputs]]
name = "dft_length"
type = "T2"
desc = "The length of the signal as a scalar. If greater than the axis dimension, the signal will be zero-padded up to `dft_length`. If less than the axis dimension, only the first `dft_length` values will be used as the signal. If not provided, the default `dft_length = signal_dim_axis`, except for the IRFFT case (`onesided=1`, `inverse=1`), in which case the default dft_length is `2 * (signal_dim_axis - 1)`."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "axis"
type = "tensor(int64)"
desc = "The axis as a scalar on which to perform the DFT. Default is `-2` (last signal axis). Negative value means counting dimensions from the back. Accepted range is $[-r, -2] \\cup [0, r-2]$ where `r = rank(input)`. The last dimension is for representing complex numbers and thus is an invalid axis."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T1"
desc = "The Fourier Transform of the input vector. For standard DFT (`onesided=0`), the output shape is: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]` (complex), with `signal_dim_axis = dft_length`. For RFFT (`onesided=1`, `inverse=0`), the output shape is: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][2]` (one-sided complex), with `signal_dim_axis = floor(dft_length/2) + 1`. For IRFFT (`onesided=1`, `inverse=1`), the output shape is: `[signal_dim0][signal_dim1][signal_dim2]...[signal_dimN][1]` (real), where `signal_dim_axis = dft_length`."

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain scalar length types to integers."
```

Computes the discrete Fourier Transform (DFT) of the input.

Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
DFT is computed and `M` denotes the conceptual "all other dimensions,"
the DFT `y[m, k]` of shape `[M, N]` is defined as

$$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$

and the inverse transform is defined as

$$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$

where $j$ is the imaginary unit.

The actual shape of the output is specified in the "output" section.

Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html
