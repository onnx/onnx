```toml
name = "RandomNormal"
domain = "ai.onnx"
node_determinism = "non_deterministic"
shape_inference = "defs::generator::utils::RandomNormalShapeInference_opset1"

[attributes.dtype]
type = "INT"
desc = "The data type for the elements of the output tensor. Default is TensorProto::FLOAT."
default = 1

[attributes.mean]
type = "FLOAT"
desc = "The mean of the normal distribution."
default = 0.0

[attributes.scale]
type = "FLOAT"
desc = "The standard deviation of the normal distribution."
default = 1.0

[attributes.seed]
type = "FLOAT"
desc = "(Optional) Seed to the random generator, if not specified we will auto generate one."

[attributes.shape]
type = "INTS"
desc = "The shape of the output tensor."
required = true

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of random values drawn from normal distribution"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain output types to float tensors."
```

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
