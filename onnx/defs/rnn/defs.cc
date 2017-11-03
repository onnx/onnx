// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;

OPERATOR_SCHEMA(SimpleRNN)
    .NumInputs(3, 6)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
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
`ReLU(X)` - max(X, 0)
`tanh(X)` - hyperbolic tangent of X
`H` - Hidden state

Equations:
  - Ht = Activation(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
)DOC")
    .Attr("activation", "The activation function for input gate. Typical "
	  "activation functions are tanh and ReLU. Default `tanh`.",
          AttrType::STRING)
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttrType::INT)
    .Attr("reverse", "Process the sequences in reverse order, default 0.",
          AttrType::INT)
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
          "in the range of [-threshold, +threshold] and is applied to the input "
          "of activations. No clip if not specified.",
          AttrType::FLOAT)
    .Input(0, "input",
           "The input sequences packed (and potentially padded) into one 3-D "
           "tensor with the shape of `[seq_length, batch_size, input_size]`.")
    .Input(1, "W", 
	   "The weight tensor for input gate. The tensor has shape "
	   "`[hidden_size, input_size]`.")
    .Input(2, "R",
	   "The recurrence weight tensor. The tensor has shape "
	   "`[hidden_size, hidden_size}`.")
    .Input(3, "B",
	   "The bias tensor for input gate. The tensor is a concatenation of"
	   "`Wbi` and `Rbi`, and has shape `[2*hidden_size]`, Optional: If not "
	   "specified - assumed to be 0.",
	   true /*optional*/)
    .Input(4, "initial_h",
	   "Optional initial value of the hidden. If not specified - assumed "
	   "to be 0. It has shape `[batch_size, hidden_size]`.",
	   true /*optional*/)	   
    .Input(5, "seq_lens",
           "Optional tensor specifying lengths of the sequences in a batch. "
           "If not specified - assumed all sequences in the batch to have "
	   "length `seq_length`. It has shape `[batch_size]`.",
	   true /*optional*/)	   
    .Output(0, "output",
	    "A tensor that concats all the intermediate output values of the "
	    "hidden. It has shape `[seq_length, batch_size, hidden_size]`.")
    .Output(1, "output_h",
            "The last output value of the hidden. It has shape "
	    "`[batch_size, hidden_size]`.");


OPERATOR_SCHEMA(GRU)
    .NumInputs(3, 6)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Computes an one-layer GRU in optimized fashion.

Notations:
`X` - input tensor
`z` - update gate
`r` - reset gate
`h` - hidden gate
`t` - time step (t-1 means previous time step)
`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
`Wb[zrh]` - W bias vectors for update, reset, and hidden gates
`Rb[zrh]` - R bias vectors for update, reset, and hidden gates
`tanh(X)` - hyperbolic tangent of X
`sigmoid(X)` - 1 / (1 + e^-X)
`H` - Hidden state

Equations (GRU with default activations):
  - zt = sigmoid(Wz*Xt + Rz*Ht-1 + Wbz + Rbz)
  - rt = sigmoid(Wr*Xt + Rr*Ht-1 + Wbr + Rbr)
  - ht = tanh(Wh*Xt + rt*(Rh*Ht-1 + Rbh) + Wbh)
  - H = (1 - zt) (.) ht + it (.) Ht-1
)DOC")
    .Attr("activations", "A list of 3 activation functions for update, reset, and "
	  "hidden gates. Typical activation functions are sigmoid and tanh. See the "
	  "equations for default if not specified.",
          AttrType::STRINGS)
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttrType::INT)
    .Attr("reverse", "Process the sequences in reverse order if 1, default 0.",
          AttrType::INT)
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
          "in the range of [-threshold, +threshold] and is applied to the input "
          "of activations. No clip if not specified.",
          AttrType::FLOAT)
    .Input(0, "input",
           "The input sequences packed (and potentially padded) into one 3-D "
           "tensor with the shape of `[seq_length, batch_size, input_size]`.")
    .Input(1, "W",
	   "The weight tensor for the gates. The weights W[zrh] are "
	   "concatenated along dimension 0. This tensor has shape "
	   "`[3 * hidden_size, input_size]`.")
    .Input(2, "R",
	   "The recurrence weight tensor. The recurrence weights R[zrh] are "
	   "concatenated along dimension 0. This tensor has shape "
	   "`[3 * hidden_size, hidden_size}`.")
    .Input(3, "B",
	   "The bias tensor for the gates. The biases Wb[zrh] and Rb[zrh] are "
	   "concatenated along dimension 0. This tensor has shape "
	   "`[6 * hidden_size]`. Optional: If not specified - assumed to be 0",
	   true /*optional*/)
    .Input(4, "initial_h",
	   "Optional initial value of the hidden. If not specified - assumed "
	   "to be 0. It has shape `[batch_size, hidden_size]`.",
	   true /*optional*/)
    .Input(5, "seq_lens",
           "Optional tensor specifying lengths of the sequences in a batch. "
           "If not specified - assumed all sequences in the batch to have "
	   "length `seq_length`. It has shape `[batch_size]`.",
	   true /*optional*/)
    .Output(0, "output",
	    "A tensor that concats all the intermediate output values of the "
	    "hidden. It has shape `[seq_length, batch_size, hidden_size]`.")	    
    .Output(1, "output_h",
            "The last output value of the hidden. It has shape "
	    "`[batch_size, hidden_size]`.");


OPERATOR_SCHEMA(LSTM)
    .NumInputs(3, 8)
    .NumOutputs(1, 2)
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
`P[iof]`  - P peephole weight matrix for input, output, and forget gates
`tanh(X)` - hyperbolic tangent of X
`sigmoid(X)` - 1 / (1 + e^-X)
`H` - Hidden state

Equations (LSTM with default activations):
  - it = sigmoid(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
  - ft = sigmoid(Wf*Xt + Rf*Ht-1 + Wbf + Rbf)
  - ot = sigmoid(Wo*Xt + Ro*Ht-1 + Wbo + Rbo)
  - ct = tanh(Wc*Xt + Rc*Ht-1 + Wbc + Rbc)
  - Ct = ft (.) Ct-1 + it (.) ct
  - H = ot (.) tanh(Ct)
)DOC")
    .Attr("activations", "A list of 4 activation functions for input, output, "
	  "forget, and cell gates. Typical activation functions are sigmoid "
	  "and tanh. See the equations for default if not specified",
          AttrType::STRINGS)
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttrType::INT)
    .Attr("reverse", "Process the sequences in reverse order if 1, default 0.",
          AttrType::INT)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
          AttrType::INT)	  
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
          "in the range of [-threshold, +threshold] and is applied to the input "
          "of activations. No clip if not specified.",
          AttrType::FLOAT)
    .Input(0, "input",
           "The input sequences packed (and potentially padded) into one 3-D "
           "tensor with the shape of `[seq_length, batch_size, input_size]`.")
    .Input(1, "W",
	   "The weight tensor for the gates. The weights W[iofc] are concatenated "
	   "along dimension 0. This tensor has shape `[4 * hidden_size, input_size]`.")
    .Input(2, "R", 
	   "The recurrence weight tensor. The recurrence weights R[iofc] are "
	   "concatenated along dimension 0. This tensor has shape "
	   "`[4 * hidden_size, hidden_size}`.")
    .Input(3, "B",
	   "The bias tensor for input gate. The biases Wb[iofc] and Rb[iofc] are "
	   "concatenated along dimension 0. This tensor has shape "
	   "`[8 * hidden_size]`. Optional: If not specified - assumed to be 0.",
	   true /*optional*/)
    .Input(4, "P",
	   "The weight tensor for peepholes. The peephole weights P[iof] are "
	   "concatenated along dimension 0. It has shape `[3 * hidde_size, hidden_size]`. "
	   "Optional: If not specified - assumed to be 0.",
	   true /*optional*/)
    .Input(5, "initial_h",
           "Optional initial value of the hidden. If not specified - assumed "
           "to be 0. It has shape `[batch_size, hidden_size]`.",
	   true /*optional*/)	   
    .Input(6, "initial_c",
           "Optional initial value of the cell. If not specified - assumed "
	   "to be 0. It has shape `[batch_size, hidden_size]`.",
	   true /*optional*/)
    .Input(7, "seq_lens",
           "Optional tensor specifying lengths of the sequences in a batch. "
           "If not specified - assumed all sequences in the batch to have "
	   "length `seq_length`. It has shape `[batch_size]`.",
	   true /*optional*/)	   
    .Output(0, "output",
	    "A tensor that concats all the intermediate output values of the "
	    "hidden. It has shape `[seq_length, batch_size, hidden_size]`.")	    
    .Output(1, "output_h",
            "The last output value of the hidden. It has shape "
	    "`[batch_size, hidden_size]`.");


OPERATOR_SCHEMA(BidirectionalLSTM)
    .NumInputs(3, 8)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Computes an one-layer bidirectional LSTM. This operator is usually
supported via some custom implementation such as CuDNN.

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
`P[iof]`  - P peephole weight matrix for input, output, and forget gates
`WR[iofc]` - WR parameter weight matrix for reverse direction input, output, forget, and cell gates
`RR[iofc]` - RR recurrence weight matrix for reverse direction input, output, forget, and cell gates
`WRb[iofc]` - WR bias vectors for reverse direction input, output, forget, and cell gates
`RRb[iofc]` - RR bias vectors for reverse direction input, output, forget, and cell gates
`PR[iof]`  - PR peephole weight matrix for reverse direction input, output, and forget gates
`tanh(X)` - hyperbolic tangent of X
`sigmoid(X)` - 1 / (1 + e^-X)
`H` - Hidden state

Equations (forward direction LSTM with default activations):
  - it = sigmoid(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
  - ft = sigmoid(Wf*Xt + Rf*Ht-1 + Wbf + Rbf)
  - ot = sigmoid(Wo*Xt + Ro*Ht-1 + Wbo + Rbo)
  - ct = tanh(Wc*Xt + Rc*Ht-1 + Wbc + Rbc)
  - Ct = ft (.) Ct-1 + it (.) ct
  - H = ot (.) tanh(Ct)
)DOC")
    .Attr("activations", "A list of 4 activation functions for input, output, "
	  "forget, and cell gates. Typical activation functions are sigmoid "
	  "and tanh. See the equations for default if not specified.",
          AttrType::STRINGS)
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttrType::INT)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
          AttrType::INT)	  
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
          "in the range of [-threshold, +threshold] and is applied to the input "
          "of activations. No clip if not specified.",
          AttrType::FLOAT)
    .Input(0, "input",
           "The input sequences packed (and potentially padded) into one 3-D "
           "tensor with the shape of `[seq_length, batch_size, input_size]`.")
    .Input(1, "W",
	   "The weight tensor for the gates. The weights W[iofc] and WR[iofc] "
	   "are concatenated along dimension 0. This tensor has shape "
	   "`[8 * hidden_size, input_size]`.")
    .Input(2, "R",
	   "The recurrence weight tensor. The recurrence weights R[iofc] "
	   "and RR[iofc] are concatenated along dimension 0. This tensor has "
	   "shape `[8 * hidden_size, hidden_size}`.")
    .Input(3, "B",
	   "The bias tensor for input gate. The biases Wb[iofc], Rb[iofc], "
	   "WRb[iofc], and RRb[iofc] are concatenated along dimension 0. "
	   "This tensor has shape `[16 * hidden_size]`. Optional: If not "
	   "specified - assumed to be 0.",
	   true /*optional*/)
    .Input(4, "P",
	   "The weight tensor for peepholes. The peephole weights P[iof] and "
	   "PR[iof] are concatenated along dimension 0. It has shape "
	   "`[6 * hidde_size, hidden_size]`. Optional: If not specified - "
	   "assumed to be 0.",
	   true /*optional*/)
    .Input(5, "initial_h",
           "Optional initial value of the hidden. If not specified - assumed "
           "to be 0. It has shape `[2, batch_size, hidden_size]`",
	   true /*optional*/)
    .Input(6, "initial_c",
           "Optional initial value of the cell. If not specified - assumed "
	   "to be 0. It has shape `[2, batch_size, hidden_size]`",
	   true /*optional*/)
    .Input(7, "seq_lens",
           "Optional tensor specifying lengths of the sequences in a batch. "
           "If not specified - assumed all sequences in the batch to have "
	   "length `seq_length`. It has shape `[batch_size]`.",
	   true /*optional*/)
    .Output(0, "output",
	    "A tensor that concats all the intermediate output values of the "
	    "hidden. It has shape `[seq_length, batch_size, hidden_size]`.")	    
    .Output(1, "output_h",
            "The last output value of the hidden. It has shape "
	    "`[batch_size, hidden_size]`.");
