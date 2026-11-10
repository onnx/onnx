```toml
name = "Adam"
domain = "ai.onnx.preview.training"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of previously accumulated gradient in running average. Default to 0.9."
default = 0.9

[attributes.beta]
type = "FLOAT"
desc = "Coefficient of previously accumulated squared-gradient in running average. Default to 0.999."
default = 0.999

[attributes.epsilon]
type = "FLOAT"
desc = "Small scalar to avoid dividing by zero."
default = 1e-06

[attributes.norm_coefficient]
type = "FLOAT"
desc = "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, which means no regularization."
default = 0.0

[attributes.norm_coefficient_post]
type = "FLOAT"
desc = "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, which means no regularization."
default = 0.0

[[inputs]]
name = "R"
type = "T1"
desc = "The initial learning rate."

[[inputs]]
name = "T"
type = "T2"
desc = "The update count of \"X\". It should be a scalar."

[[inputs]]
name = "inputs"
type = "T3"
desc = "The tensors to be optimized, followed by their respective gradients, followed by their respective accumulated gradients (aka momentum), followed by their respective accumulated squared gradients. For example, to optimize tensors \"X_1\" and \"X_2,\", the input list would be [\"X_1\", \"X_2\", gradient of \"X_1\", gradient of \"X_2\", accumulated gradient of \"X_1\", accumulated gradient of \"X_2\", accumulated squared gradient of \"X_1\", accumulated squared gradient of \"X_2\"]."
option = "variadic"
homogeneous = false

[[outputs]]
name = "outputs"
type = "T3"
desc = "New values of optimized tensors, followed by their respective new accumulated gradients, followed by their respective new accumulated squared gradients. For example, if two tensors \"X_1\" and \"X_2\" are optimized, the outputs list would be [new value of \"X_1\", new value of \"X_2\", new accumulated gradient of \"X_1\", new accumulated gradient of \"X_2\", new accumulated squared gradient of \"X_1\", new accumulated squared gradient of \"X_2\"]."
option = "variadic"
homogeneous = false

[type_constraints.T1]
types = ["tensor(float)", "tensor(double)"]
desc = "Constrain input types to float scalars."

[type_constraints.T2]
types = ["tensor(int64)"]
desc = "Constrain input types to 64-bit integer scalars."

[type_constraints.T3]
types = ["tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Compute one iteration of Adam, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. First of all, Adam requires
some parameters:

 - The learning-rate "R".
 - The update count "T". That is, the number of training iterations conducted.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A small constant "epsilon" to avoid dividing-by-zero.
 - Two coefficients, "alpha" and "beta".

At each Adam iteration, the optimized tensors are moved along a direction
computed based on their exponentially-averaged historical gradient and
exponentially-averaged historical squared gradient. Assume that only a tensor
"X" is being optimized. The rest of required information is

 - the value of "X",
 - "X"'s gradient (denoted by "G"),
 - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
 - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

Some of those parameters are passed into this operator as input tensors and others
are stored as this operator's attributes. Specifically, this operator's input tensor
list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
the second input, and so on. Other parameters are given as attributes because they
are constants. Moreover, the corresponding output tensors are

 - the new value of "X" (called "X_new"),
 - the new exponentially-averaged historical gradient (denoted by "V_new"), and
 - the new exponentially-averaged historical squared gradient (denoted by "H_new").

Those outputs are computed following the pseudo code below.

Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
numpy-style broadcasting support. The pseudo code to compute those outputs is:

  // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
  G_regularized = norm_coefficient * X + G

  // Update exponentially-averaged historical gradient.
  V_new = alpha * V + (1 - alpha) * G_regularized

  // Update exponentially-averaged historical squared gradient.
  H_new = beta * H + (1 - beta) * G_regularized * G_regularized

  // Compute the element-wise square-root of H_new. V_new will be element-wisely
  // divided by H_sqrt for a better update direction.
  H_sqrt = Sqrt(H_new) + epsilon

  // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
  R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

  // Compute new value of "X".
  X_new = X - R_adjusted * V_new / H_sqrt

  // Post-update regularization.
  X_final = (1 - norm_coefficient_post) * X_new

If there are multiple inputs to be optimized, the pseudo code will be applied
independently to each of them.
