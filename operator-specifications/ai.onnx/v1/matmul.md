```toml
name = "MatMul"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MatMulShapeInference_opset1"

[[inputs]]
name = "A"
type = "T"
desc = "N-dimensional matrix A"

[[inputs]]
name = "B"
type = "T"
desc = "N-dimensional matrix B"

[[outputs]]
name = "Y"
type = "T"
desc = "Matrix multiply results from A * B"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
