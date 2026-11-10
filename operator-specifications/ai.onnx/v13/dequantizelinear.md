```toml
name = "DequantizeLinear"
domain = "ai.onnx"
shape_inference = "defs::quantization::utils::DequantizeLinearShapeInference_opset13"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."
default = 1

[[inputs]]
name = "x"
type = "T"
desc = "N-D quantized input tensor to be de-quantized."

[[inputs]]
name = "x_scale"
type = "tensor(float)"
desc = "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, or a 1-D tensor for per-axis dequantization."

[[inputs]]
name = "x_zero_point"
type = "T"
desc = "Zero point for input 'x'. Shape must match x_scale. It's optional. Zero point is 0 when it's not specified."
option = "optional"
```

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
