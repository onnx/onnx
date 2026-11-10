```toml
name = "Concat"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ConcatShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "Which axis to concat on. A negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(inputs).."
required = true
```

Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
