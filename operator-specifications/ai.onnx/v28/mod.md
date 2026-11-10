```toml
name = "Mod"
domain = "ai.onnx"

[attributes.fmod]
type = "INT"
desc = "Whether the operator should use floor (0) or truncation (1) to calculate the quotient."
default = 0

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to numeric tensors."
```

Performs an element-wise binary modulo operation.
The `fmod` attribute determines how the quotient is rounded. Its value must be
`0` (default) or `1`.

If `fmod` is `0`, the output is calculated as `A - floor(A / B) * B`.
The result has the same sign as `B`.
For floating-point inputs, the following special cases apply:
- If `x` is `±0` and `y` is nonzero, `±0` with the sign of `y` is returned.
- If `x` is `±∞` and `y` is not `NaN`, `NaN` is returned.
- If `y` is `±0` and `x` is not `NaN`, `NaN` is returned.
- If `y` is `±∞` and `x` is finite and nonzero, `x` is returned when `x` and
  `y` have the same sign; otherwise, `y` is returned.
- If either argument is `NaN`, `NaN` is returned.

If `fmod` is `1`, the output is calculated as `A - trunc(A / B) * B`.
The result has the same sign as `A`, except that either signed zero may be
returned when `A` is `-0` and `B` is positive. For floating-point inputs,
the following special cases apply:
- If `x` is `-0` and `y` is greater than zero, either `+0` or `-0` may be returned.
- If `x` is `±∞` and `y` is not `NaN`, `NaN` is returned.
- If `y` is `±0` and `x` is not `NaN`, `NaN` should be returned.
- If `y` is `±∞` and `x` is finite, `x` is returned.
- If either argument is `NaN`, `NaN` is returned.

This operator supports **multidirectional (i.e., NumPy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
