```toml
name = "MelWeightMatrix"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MelWeightMatrixShapeInference_opset17"

[attributes.output_datatype]
type = "INT"
desc = "The data type of the output tensor. Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T3. The default value is 1 = FLOAT. "
default = 1

[[inputs]]
name = "num_mel_bins"
type = "T1"
desc = "The number of bands in the mel spectrum."
differentiation = "non_differentiable"

[[inputs]]
name = "dft_length"
type = "T1"
desc = "The size of the original DFT. The size of the original DFT is used to infer the size of the onesided DFT, which is understood to be floor(dft_length/2) + 1, i.e. the spectrogram only contains the nonredundant DFT bins."
differentiation = "non_differentiable"

[[inputs]]
name = "sample_rate"
type = "T1"
desc = "Samples per second of the input signal used to create the spectrogram. Used to figure out the frequencies corresponding to each spectrogram bin, which dictates how they are mapped into the mel scale."
differentiation = "non_differentiable"

[[inputs]]
name = "lower_edge_hertz"
type = "T2"
desc = "Lower bound on the frequencies to be included in the mel spectrum. This corresponds to the lower edge of the lowest triangular band."
differentiation = "non_differentiable"

[[inputs]]
name = "upper_edge_hertz"
type = "T2"
desc = "The desired top edge of the highest frequency band."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T3"
desc = "The Mel Weight Matrix. The output has the shape: [floor(dft_length/2) + 1][num_mel_bins]."
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain to integer tensors."

[type_constraints.T2]
types = "all_float_types_ir4"
desc = "Constrain to float tensors"

[type_constraints.T3]
types = "all_numeric_types_ir4"
desc = "Constrain to any numerical types."
```

Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" M of shape [frames, num_mel_bins].
