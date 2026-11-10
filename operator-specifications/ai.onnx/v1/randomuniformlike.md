```toml
name = "RandomUniformLike"
domain = "ai.onnx"
node_determinism = "non_deterministic"
shape_inference = "defs::generator::utils::RandomUniformLikeShapeInference_opset1"

[attributes.dtype]
type = "INT"
desc = "(Optional) The data type for the elements of the output tensor, if not specified, we will use the data type of the input tensor."

[attributes.high]
type = "FLOAT"
desc = "Upper boundary of the output values."
default = 1.0

[attributes.low]
type = "FLOAT"
desc = "Lower boundary of the output values."
default = 0.0

[attributes.seed]
type = "FLOAT"
desc = "(Optional) Seed to the random generator, if not specified we will auto generate one."

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor to copy shape and optionally type information from."

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor of random values drawn from uniform distribution"

[type_constraints.T1]
types = "all_tensor_types"
desc = "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type."

[type_constraints.T2]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain output types to float tensors."
```

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
