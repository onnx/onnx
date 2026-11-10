```toml
name = "BatchNormalization"
domain = "ai.onnx"
num_outputs = [1, 3]
shape_inference = "defs::nn::utils::BatchNormalizationShapeInference_opset14"

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.momentum]
type = "FLOAT"
desc = "Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum)."
default = 0.9

[attributes.training_mode]
type = "INT"
desc = "If set to true, it indicates BatchNormalization is being used for training, and outputs 1, 2, 3, and 4 would be populated."
default = 0

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1"
differentiation = "differentiable"

[[inputs]]
name = "scale"
type = "T"
desc = "Scale tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Bias tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "input_mean"
type = "U"
desc = "running (training) or estimated (testing) mean tensor of shape (C)."
differentiation = "differentiable"

[[inputs]]
name = "input_var"
type = "U"
desc = "running (training) or estimated (testing) variance tensor of shape (C)."
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "The output tensor of the same shape as X"
differentiation = "differentiable"

[[outputs]]
name = "running_mean"
type = "U"
desc = "The running mean after the BatchNormalization operator."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "running_var"
type = "U"
desc = "The running variance after the BatchNormalization operator. This op uses the population size (N) for calculating variance, and not the sample size N-1."
option = "optional"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[type_constraints.U]
types = "all_float_types_ir4"
desc = "Constrain mean and variance types to float tensors. It allows all float type for U."
```

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

Output case #1: Y, running_mean, running_var (training_mode=True)
Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

where:

current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)

Notice that ReduceVar refers to the population variance, and it equals to
sum(sqrd(x_i - x_avg)) / N
where N is the population size (this formula does not use sample size N - 1).

```

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
