```toml
name = "Squeeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SqueezeShapeInference_opset11"

[attributes.axes]
type = "INTS"
desc = "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data)."
```
