```toml
name = "Div"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MathShapeInference"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numeric tensors."
```

Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

For integer inputs, the result is computed using truncating division (rounding toward zero). For example, `-11 / 3` yields `-3`.
(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
