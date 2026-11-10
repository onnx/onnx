```toml
name = "Sub"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MathShapeInference"
data_propagation = "defs::math::utils::SubDataPropagation_opset14"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numeric tensors."
```

Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
