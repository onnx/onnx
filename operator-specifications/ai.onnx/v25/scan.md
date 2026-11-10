```toml
name = "Scan"
domain = "ai.onnx"
shape_inference = "defs::controlflow::utils::ScanInferenceFunction"

[type_constraints.V]
types = "all_tensor_types_ir13"
desc = "All Tensor types up to IRv13."
```
