#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> RNNDocGeneratorOld(const char* name) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("foward"));
    schema.Attr(
        "hidden_size",
        "Number of neurons in the hidden layer",
        AttributeProto::INT,
        OPTIONAL);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM.",
        AttributeProto::FLOATS,
        OPTIONAL);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM.",
        AttributeProto::FLOATS,
        OPTIONAL);
    schema.Attr(
        "output_sequence",
        "The sequence output for the hidden is optional if 0. Default 0.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T");
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
        "It is optional if `output_sequence` is 0.",
        "T",
        OpSchema::Optional);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeConstraint(
        "T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
  };
}


ONNX_OPERATOR_SCHEMA(GRU)
    .SetDoc(R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

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

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

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

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC")
    .Attr(
        "activations",
        "A list of 2 (or 4 if bidirectional) activation functions "
        "for update, reset, and hidden gates. The activation functions must be one "
        "of the activation functions specified above. Optional: See the equations "
        "for default if not specified.",
        AttributeProto::STRINGS,
        OPTIONAL)
    .Attr(
        "linear_before_reset",
        "When computing the output of the hidden gate, "
        "apply the linear transformation before multiplying by the output of the "
        "reset gate.",
        AttributeProto::INT,
        OPTIONAL)
    .Input(
        1,
        "W",
        "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
        "(if bidirectional) along dimension 0. This tensor has shape "
        "`[num_directions, 3*hidden_size, input_size]`.",
        "T")
    .Input(
        2,
        "R",
        "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
        "(if bidirectional) along dimension 0. This tensor has shape "
        "`[num_directions, 3*hidden_size, hidden_size]`.",
        "T")
    .Input(
        3,
        "B",
        "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
        "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
        "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
        "- assumed to be 0",
        "T",
        OpSchema::Optional)
    .FillUsing(RNNDocGeneratorOld("GRU"));
} // namespace ONNX_NAMESPACE
