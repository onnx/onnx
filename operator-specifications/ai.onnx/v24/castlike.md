```toml
name = "CastLike"
domain = "ai.onnx"

[attributes.round_mode]
type = "STRING"
desc = "Rounding mode for conversion to float8e8m0. It only applies to casting to float8e8m0 and is `up` by default. `up`: round to nearest value away from zero, `down`: round to nearest value towards zero, `nearest`: round to nearest value and ties round up. Please refer to operator Cast description for further details."
default = "up"

[attributes.saturate]
type = "INT"
desc = "The parameter defines how the conversion behaves if an input value is out of range of the destination type. It only applies for float 8 conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default. Please refer to operator Cast description for further details."
default = 1

[type_constraints.T1]
types = "all_non_complex_tensor_types_ir12"
desc = "Constrain input types. Casting from complex is not supported."

[type_constraints.T2]
types = "all_non_complex_tensor_types_ir12"
desc = "Constrain output types. Casting to complex is not supported."
```
