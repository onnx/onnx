```toml
name = "Split"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SplitShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input)."
default = 0

[attributes.split]
type = "INTS"
desc = "length of each output. Values should be >= 0."
```
