```toml
name = "QuantizeLinear"
domain = "ai.onnx"

[[inputs]]
name = "x"
type = "T1"
desc = "N-D full precision Input tensor to be quantized."

[[inputs]]
name = "y_scale"
type = "tensor(float)"
desc = "Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "y_zero_point"
type = "T2"
desc = "Zero point for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization. Default value is uint8 typed 0 if it's not specified."
option = "optional"

[[outputs]]
name = "y"
type = "T2"
desc = "N-D quantized output tensor. It has same shape as input 'x'."

[type_constraints.T1]
types = ["tensor(float)", "tensor(int32)"]
desc = "Constrain 'x' to float or int32 tensor."

[type_constraints.T2]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor."
```

The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
