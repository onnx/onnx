```toml
name = "RNN"
domain = "ai.onnx"
shape_inference = "defs::rnn::utils::RNNShapeInference_opset7_to_13"

[attributes.activation_alpha]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01."

[attributes.activation_beta]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators."

[attributes.activations]
type = "STRINGS"
desc = "One (or two if bidirectional) activation function for input gate. The activation function must be one of the activation functions specified above. Optional: Default `Tanh` if not specified."
default = ["Tanh", "Tanh"]

[attributes.clip]
type = "FLOAT"
desc = "Cell clip threshold. Clipping bounds the elements of a tensor in the range of [-threshold, +threshold] and is applied to the input of activations. No clip if not specified."

[attributes.direction]
type = "STRING"
desc = "Specify if the RNN is forward, reverse, or bidirectional. Must be one of forward (default), reverse, or bidirectional."
default = "forward"

[attributes.hidden_size]
type = "INT"
desc = "Number of neurons in the hidden layer"

[[outputs]]
name = "Y"
type = "T"
desc = "A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
option = "optional"

[[outputs]]
name = "Y_h"
type = "T"
desc = "The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
```

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
