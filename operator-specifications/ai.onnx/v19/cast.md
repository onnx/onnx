```toml
name = "Cast"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::CastShapeInference_opset19"
data_propagation = "defs::tensor::utils::CastDataPropagation_opset19"

[attributes.saturate]
type = "INT"
desc = "The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. All cases are fully described in two tables inserted in the operator description."
default = 1

[attributes.to]
type = "INT"
desc = "The data type to which the elements of the input tensor are cast. Strictly must be one of the types from DataType enum in TensorProto"
required = true

[type_constraints.T1]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
    "tensor(string)",
    "tensor(bfloat16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain input types. Casting from complex is not supported."

[type_constraints.T2]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
    "tensor(string)",
    "tensor(bfloat16)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "Constrain output types. Casting to complex is not supported."
```

The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules
if the destination type is not a float 8 type.

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.

Float 8 type were introduced to speed up the training of
deep models. By default the conversion of a float *x* obeys
to the following rules. `[x]` means the value rounded to
the target mantissa width.

| x                 | E4M3FN   | E4M3FNUZ | E5M2     | E5M2FNUZ |
| ----------------- | -------- | -------- | -------- | -------- |
| 0                 | 0        | 0        | 0        | 0        |
| -0                | -0       | 0        | -0       | 0        |
| NaN               | NaN      | NaN      | NaN      | NaN      |
| Inf               | FLT_MAX  | NaN      | FLT_MAX  | NaN      |
| -Inf              | -FLT_MAX | NaN      | -FLT_MAX | NaN      |
| \[x\] > FLT_MAX   | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
| \[x\] \< -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else              | RNE      | RNE      | RNE      | RNE      |

The behavior changes if the parameter 'saturate' is set to False.
The rules then become:

| x                 | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
| ----------------- | ------ | -------- | ---- | -------- |
| 0                 | 0      | 0        | 0    | 0        |
| -0                | -0     | 0        | -0   | 0        |
| NaN               | NaN    | NaN      | NaN  | NaN      |
| -NaN              | -NaN   | NaN      | -NaN | NaN      |
| Inf               | NaN    | NaN      | Inf  | NaN      |
| -Inf              | -NaN   | NaN      | -Inf | NaN      |
| \[x\] > FLT_MAX   | NaN    | NaN      | Inf  | NaN      |
| \[x\] \< -FLT_MAX | NaN    | NaN      | -Inf | NaN      |
| else              | RNE    | RNE      | RNE  | RNE      |
