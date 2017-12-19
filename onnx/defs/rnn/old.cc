// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

namespace onnx {

std::function<void(OpSchema&)> RNNDocGenerator(const char* name);

OPERATOR_SCHEMA(LSTM)
    .SinceVersion(1)
    .SetDoc(R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

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

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
)DOC")
    .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
          "for input, output, forget, cell, and hidden. The activation functions must "
          "be one of the activation functions specified above. Optional: See the equations "
          "for default if not specified.",
          AttributeProto::STRINGS)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
          AttributeProto::INT)
    .Input(1, "W",
	   "The weight tensor for the gates. Concatenation of `W[iofc]` and "
           "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
	   "`[num_directions, 4*hidden_size, input_size]`.", "T")
    .Input(2, "R",
	   "The recurrence weight tensor. Concatenation of `R[iofc]` and "
	   "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
           "`[num_directions, 4*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
	   "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
	   "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
           "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
	   "specified - assumed to be 0.", "T",
       OpSchema::Optional)
    .Input(6, "initial_c",
           "Optional initial value of the cell. If not specified - assumed "
	   "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
	   "T", OpSchema::Optional)
    .Input(7, "P",
	   "The weight tensor for peepholes. Concatenation of `P[iof]` and "
	   "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
	   "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
	   "assumed to be 0.", "T",
       OpSchema::Optional)
    .FillUsing(RNNDocGenerator("LSTM"));
}  // namespace onnx
