```toml
name = "Dropout"
domain = "ai.onnx"
node_determinism = "non_deterministic"
shape_inference = false

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[attributes.is_test]
type = "INT"
desc = "(int, default 0) if nonzero, run dropout in test mode where the output is simply Y = X."
default = 0

[attributes.ratio]
type = "FLOAT"
desc = "(float, default 0.5) the ratio of random dropout"
default = 0.5

[[inputs]]
name = "data"
type = "T"
desc = "The input data as Tensor."

[[outputs]]
name = "output"
type = "T"
desc = "The output."

[[outputs]]
name = "mask"
type = "T"
desc = "The output mask. If is_test is nonzero, this output is not filled."
option = "optional"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
