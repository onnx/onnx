```toml
name = "Flatten"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n). "
default = 1
```
