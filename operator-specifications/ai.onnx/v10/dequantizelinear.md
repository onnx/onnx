```toml
name = "DequantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::DequantizeLinearShapeInference_opset10"

[[inputs]]
name = "x"
type = "T"
desc = "N-D quantized input tensor to be de-quantized."

[[inputs]]
name = "x_scale"
type = "tensor(float)"
desc = "Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "x_zero_point"
type = "T"
desc = "Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. It's optional. 0 is the default value when it's not specified."
option = "optional"

[[outputs]]
name = "y"
type = "tensor(float)"
desc = "N-D full precision output tensor. It has same shape as input 'x'."

[type_constraints.T]
types = ["tensor(int8)", "tensor(uint8)", "tensor(int32)"]
desc = "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor."
```

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
