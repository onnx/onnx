```toml
name = "BatchNormalization"
domain = "ai.onnx"
num_outputs = [1, 5]

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."
required = true

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero, default is 1e-5f."
default = 1e-05

[attributes.is_test]
type = "INT"
desc = "If set to nonzero, run spatial batch normalization in test mode, default is 0."
default = 0

[attributes.momentum]
type = "FLOAT"
desc = "Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f."
default = 0.9

[attributes.spatial]
type = "INT"
desc = "If true, compute the mean and variance across all spatial elements If false, compute the mean and variance across per feature.Default is 1."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "The input 4-dimensional tensor of shape NCHW."

[[inputs]]
name = "scale"
type = "T"
desc = "The scale as a 1-dimensional tensor of size C to be applied to the output."

[[inputs]]
name = "B"
type = "T"
desc = "The bias as a 1-dimensional tensor of size C to be applied to the output."

[[inputs]]
name = "mean"
type = "T"
desc = "The running mean (training) or the estimated mean (testing) as a 1-dimensional tensor of size C."

[[inputs]]
name = "var"
type = "T"
desc = "The running variance (training) or the estimated variance (testing) as a 1-dimensional tensor of size C."

[[outputs]]
name = "Y"
type = "T"
desc = "The output 4-dimensional tensor of the same shape as X."

[[outputs]]
name = "mean"
type = "T"
desc = "The running mean after the BatchNormalization operator. Must be in-place with the input mean. Should not be used for testing."
option = "optional"

[[outputs]]
name = "var"
type = "T"
desc = "The running variance after the BatchNormalization operator. Must be in-place with the input var. Should not be used for testing."
option = "optional"

[[outputs]]
name = "saved_mean"
type = "T"
desc = "Saved mean used during training to speed up gradient computation. Should not be used for testing."
option = "optional"

[[outputs]]
name = "saved_var"
type = "T"
desc = "Saved variance used during training to speed up gradient computation. Should not be used for testing."
option = "optional"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
