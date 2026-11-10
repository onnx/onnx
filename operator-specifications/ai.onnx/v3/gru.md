```toml
name = "GRU"
domain = "ai.onnx"

[attributes.activation_alpha]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01."

[attributes.activation_beta]
type = "FLOATS"
desc = "Optional scaling values used by some activation functions. The values are consumed in the order of activation functions, for example (f, g, h) in LSTM. Default values are the same as of corresponding ONNX operators."

[attributes.activations]
type = "STRINGS"
desc = "A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates. The activation functions must be one of the activation functions specified above. Optional: See the equations for default if not specified."

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

[attributes.linear_before_reset]
type = "INT"
desc = "When computing the output of the hidden gate, apply the linear transformation before multiplying by the output of the reset gate."
default = 0

[attributes.output_sequence]
type = "INT"
desc = "The sequence output for the hidden is optional if 0. Default 0."
default = 0

[[outputs]]
name = "Y"
type = "T"
desc = "A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`. It is optional if `output_sequence` is 0."
option = "optional"

[[outputs]]
name = "Y_h"
type = "T"
desc = "The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`."
option = "optional"
```
