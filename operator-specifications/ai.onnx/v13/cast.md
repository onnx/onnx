```toml
name = "Cast"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor to be cast."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor with the same shape as input with type specified by the 'to' argument"
differentiation = "differentiable"

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

In more detail, the conversion among numerical types should follow these rules:

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
