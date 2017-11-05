// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;
using namespace onnx;

// CuDNN parameters not included yet:
// - dropout (as we primarily target inference)
// Description below is borrowed from CuDNN and TensorRT docs

OPERATOR_SCHEMA(OptimizedRNN)
    .NumInputs(2, 4)
    .NumOutputs(1, 3)
    .SetDoc(R"DOC(
Computes a stack of several RNNs in optimized fashion. This operator is usually
implemented via CuDNN and thus most of the attributes and weights layout matches
directly.
)DOC")
    .Attr("cell_type", R"DOC(
Types of the cell: `relu`, `tanh`, `gru`, `lstm`

Equation definitions:
`i` - input gate
`o` - output gate
`f` - forget gate
`z` - update gate
`r` - reset gate
`c` - cell gate
`h` - hidden gate
`t` - time step (t-1 means previous time step)
`Xi` - input tensor
`W[izrfcoh]` - W parameter weight matrices for the corresponding gates
`R[izrfcoh]` - R parameter weight matrices for the corresponding gates
`Wb[izrfcoh]` - W parameter bias vectors for the corresponding gates
`Rb[izrfcoh]` - R parameter bias vectors for the corresponding gates
`ReLU(X)` - max(X, 0)
`tanh` - hyperbolic tangent of X
`sigmoid(X)` - 1 / (1 + e^-X)
`[C|H]` - Cell/Hidden state

- Equations:
  `relu`
  - Ht = ReLU(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
  `tanh`
  - Ht = tanh(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
  `lstm`
  - it = sigmoid(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
  - ft = sigmoid(Wf*Xt + Rf*Ht-1 + Wbf + Rbf)
  - ot = sigmoid(Wo*Xt + Ro*Ht-1 + Wbo + Rbo)
  - ct = tanh(Wc*Xt + Rc*Ht-1 + Wbc + Rbc)
  - C = ft * Ct-1 + it * ct
  - H = ot * tanh(C)
  `gru`
  - zt = sigmoid(Wz*Xt + Rz*Ht-1 + Wbz + Rbz)
  - rt = sigmoid(Wr*Xt + Rr*Ht-1 + Wbr + Rbr)
  - ht = tanh(Wh*Xt + rt *(Rh*Ht-1 + Rbh) + Wbh)
  - H = (1 - zt) * ht + it * Ht-1

Note, that for LSTM and 2 out of 3 gates for GRU, there are duplicate biases for
the gates (model is overparametrized). It follows CuDNN/TensorRT convention and
allows to make spec more uniform.
)DOC", AttrType::STRING, true)
    .Attr("directions",
          "Number of directions: 1 for unidirectional (default) and 2 for "
          "bidirectional",
          AttrType::INT)
    .Attr("skip_input_transform",
          "If set, skips linear transformation on the input of the first layer",
          AttrType::INT)
    .Attr("num_layers", "Numbers of RNN layers in the stack, default 1",
          AttrType::INT)
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttrType::INT)
    .Input(0, "weights", R"DOC(
All parameters of the stack packed together in the opaque tensor. The size must
be compatible with input attributes passed to the op.

The layout format is the one used by CuDNN and very similar to TensorRT:

The weight structure holds weights and biases for each layer of the network.
Each parameter matrix is linearly appended after the previous parameter matrix
without padding.

The order of matrixes `{K, L, D, R, N, C}` is defined as:
 - K - type of the matrix: `weight` (first) or `bias` second
 - L - The number of layers in the RNN - `num_layers`
 - D - The direction of the layer: normal (first) or reverse (second).
                                   (in case of `directions=2`)
 - R - The type of the connection: `input-hidden` (first) or
                                   `hidden-hidden` (second)
 - N - The number of gates matrices in the RNN, dependent on the `cell_type`:
 -- For `relu` or `tanh` there is one gate
 -- For `gru` there are 3 gates ordered as `reset`, `update`, `hidden`
 -- For `lstm` there are 4 gates ordered as `input`, `forget`, `cell`, `output`
 - C - The size of each matrix, which varies.
 -- If the linear layer on the input is skipped (`skip_input_transform=1`)
    and then for the first layer (`L=1`) the weight matrix (`K=weight`)
    on the input connection (`R=input-hidden`) is skipped,
    i.e. has 0 parameters in the list
 -- For the first layer (`L=1`) weight matrix (`K=weight`) on input connection
    (`R=input-hidden`), dimensions are `{hidden_size, input_size}`
 -- For other layers (`L>1`) weight matrix (`K=weight`) on input connection
    (`R=input-hidden`), dimensions are `{hidden_size, directions * hidden_size}`
 -- For weight matrix (`K=weight`) on recurrent connection (`R=hidden-hidden`),
    dimensions are `{hidden_size, hidden_size}`
 -- For all biases (`K=bias`), dimensions are `{hidden_size}`
)DOC", "T")
    .Input(1, "input",
           "The input sequences packed (and potentially padded) into one 3-D "
           "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    // TODO: do we want to allow different lengths of sequences in a minibatch?
    // CuDNN supports it, but not all backend implementations do. One way to
    // encode would be int-valued tensor denoting lengths of each sequence in
    // the batch.
    .Input(2, "initial_h",
           "Optional initial value of the hidden. If not specified - assumed "
           "to be 0. Dimensions `[num_layers * directions, batch_size, "
           "hidden_size]`", "T")
    .Input(3, "initial_c",
           "For LSTM only: optional initial value of the cell. If not "
           "specified - assumed to be 0. Dimensions `[num_layers * directions, "
           "batch_size, hidden_size]`", "T")
    .Output(0, "output", "The output 3-dim sequence.", "T")
    .Output(1, "output_h",
            "Optional output value of the hidden. Same shape as input_h", "T")
    .Output(2, "output_c",
            "For LSTM only: optional output value of the cell. Same shape as "
            "input_h", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.");
