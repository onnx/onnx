```toml
name = "QuantizeLinear"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis and blocked quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, rendering the axis unnecessary in this scenario."
default = 1

[attributes.block_size]
type = "INT"
desc = "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, `y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is `[ceil(Di/Si), ceil(Di/(Si-1))-1]`"
default = 0

[attributes.output_dtype]
type = "INT"
desc = "(Optional) The output data type. If not supplied, the output data type is inferred from `y_zero_point` data type (`T3`). If neither `output_dtype` nor `y_zero_point` are supplied, output data type is uint8. If both `output_dtype` and `y_zero_point` are specified, `output_dtype` must be `T3`."
default = 0

[attributes.precision]
type = "INT"
desc = "(Optional) The precision of the division operation between `x` and `y_scale`. If not provided, it will be the same as the type of `y_scale`."
default = 0

[attributes.saturate]
type = "INT"
desc = "The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 quantization (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. All cases are fully described in two tables inserted in the operator description. It has no effect for float4e2m1, float6e2m3, or float6e3m2, since those types have no non-saturating (infinity-representable) encoding to fall back to."
default = 1

[type_constraints.T1]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"]
desc = "The type of the input 'x'."

[type_constraints.T2]
types = [
    "tensor(float)",
    "tensor(float16)",
    "tensor(bfloat16)",
    "tensor(int32)",
    "tensor(float8e8m0)",
]
desc = "The type of the input 'y_scale'."

[type_constraints.T3]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(int16)",
    "tensor(uint16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
    "tensor(uint4)",
    "tensor(int4)",
    "tensor(float4e2m1)",
    "tensor(uint2)",
    "tensor(int2)",
    "tensor(float6e2m3)",
    "tensor(float6e3m2)",
]
desc = "The type of the input `y_zero_point` and the output `y`."
```
