```toml
name = "LSTM"
domain = "ai.onnx"

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
desc = "The weight tensor for peepholes. Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. It has shape `[num_directions, 3*hidden_size]`. Optional: If not specified - assumed to be 0."
option = "optional"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[type_constraints.T1]
types = ["tensor(int32)"]
desc = "Constrain seq_lens to integer tensor."
```
