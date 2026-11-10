```toml
name = "LSTM"
domain = "ai.onnx"

[attributes.activation_alpha]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01."

[attributes.activation_beta]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators."

[attributes.activations]
type = "STRINGS"
desc = "A list of 3 (or 6 if bidirectional) activation functions for input, output, forget, cell, and hidden. The activation functions must be one of the activation functions specified above. Optional: See the equations for default if not specified."

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

[attributes.input_forget]
type = "INT"
desc = "Couple the input and forget gates if 1."
default = 0

[attributes.layout]
type = "INT"
desc = "The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c. If 0, the following shapes are expected: X.shape = [seq_length, batch_size, input_size], Y.shape = [seq_length, num_directions, batch_size, hidden_size], initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size]. If 1, the following shapes are expected: X.shape = [batch_size, seq_length, input_size], Y.shape = [batch_size, seq_length, num_directions, hidden_size], initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [batch_size, num_directions, hidden_size]."
default = 0

[[inputs]]
name = "X"
type = "T"
desc = "The input sequences packed (and potentially padded) into one 3-D tensor with the shape of `[seq_length, batch_size, input_size]`."
differentiation = "differentiable"

[[inputs]]
name = "W"
type = "T"
desc = "The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`."
differentiation = "differentiable"

[[inputs]]
name = "R"
type = "T"
desc = "The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not specified - assumed to be 0."
option = "optional"
differentiation = "differentiable"

[[inputs]]
name = "sequence_lens"
type = "T1"
desc = "Optional tensor specifying lengths of the sequences in a batch. If not specified - assumed all sequences in the batch to have length `seq_length`. It has shape `[batch_size]`."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "initial_h"
type = "T"
desc = "Optional initial value of the hidden. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "initial_c"
type = "T"
desc = "Optional initial value of the cell. If not specified - assumed to be 0. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "P"
type = "T"
desc = "The weight tensor for peepholes. Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0."
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "Y_h"
type = "T"
desc = "The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "Y_c"
type = "T"
desc = "The last output value of the cell. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
differentiation = "differentiable"
```

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
