```toml
name = "QuantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::QuantizeLinearShapeInference_opset13"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."
default = 1

[[inputs]]
name = "x"
type = "T1"
desc = "N-D full precision Input tensor to be quantized."

[[inputs]]
name = "y_scale"
type = "tensor(float)"
desc = "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, or a 1-D Tensor for per-axis quantization."

[[inputs]]
name = "y_zero_point"
type = "T2"
desc = "Zero point for doing quantization to get 'y'. Shape must match y_scale. Default is uint8 with zero point of 0 if it's not specified."
option = "optional"
```

The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
