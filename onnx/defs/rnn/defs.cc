/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"
#include "onnx/defs/doc_strings.h"

namespace ONNX_NAMESPACE {
ONNX_API void RNNShapeInference(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  auto layout_value = getAttribute(ctx, "layout", 0);

  if (hasInputShape(ctx, 0)) {
    const auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    }
    seq_length = first_input_shape.dim((layout_value == 0) ? 0 : 1);
    batch_size = first_input_shape.dim((layout_value == 0) ? 1 : 0);
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);

    if (layout_value == 0) {
      auto dims = {seq_length, num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 0, dims);
    } else {
      auto dims = {batch_size, seq_length, num_directions, hidden_size};
      updateOutputShape(ctx, 0, dims);
    }
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 1, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 1, dims);
    }
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

    if (layout_value == 0) {
      auto dims = {num_directions, batch_size, hidden_size};
      updateOutputShape(ctx, 2, dims);
    } else {
      auto dims = {batch_size, num_directions, hidden_size};
      updateOutputShape(ctx, 2, dims);
    }
  }
}

static std::function<void(OpSchema&)> RNNDocGenerator(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("forward"));
    schema.Attr(
        "layout",
        "The shape format of inputs X, initial_h and outputs Y, Y_h. "
        "If 0, the following shapes are expected: "
        "X.shape = [seq_length, batch_size, input_size], "
        "Y.shape = [seq_length, num_directions, batch_size, hidden_size], "
        "initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size]. "
        "If 1, the following shapes are expected: "
        "X.shape = [batch_size, seq_length, input_size], "
        "Y.shape = [batch_size, seq_length, num_directions, hidden_size], "
        "initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators."
        "For example with LeakyRelu, the default alpha is 0.01.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeConstraint("T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
    schema.TypeAndShapeInferenceFunction(RNNShapeInference);
  };
}

static const char* RNN_ver22_doc = kDoc_e49986c33e73;

ONNX_OPERATOR_SET_SCHEMA(
    RNN,
    22,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(RNN_ver22_doc) + GenerateOptionalArgumentsDoc()))
        .Attr(
            "activations",
            "One (or two if bidirectional) activation function for "
            "input gate. The activation function must be one of the activation "
            "functions specified above. Optional: Default `Tanh` if not specified.",
            AttributeProto::STRINGS,
            std::vector<std::string>{"Tanh", "Tanh"})
        .Input(
            1,
            "W",
            "The weight tensor for input gate. Concatenation of `Wi` and `WBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, input_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `Ri` and `RBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, hidden_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` "
            "and `[WBbi, RBbi]` (if bidirectional). The tensor has shape "
            "`[num_directions, 2*hidden_size]`. Optional: If not specified - assumed "
            "to be 0.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .FillUsing(RNNDocGenerator("RNN")));

static const char* GRU_ver22_doc = kDoc_02528396a05f;

ONNX_OPERATOR_SET_SCHEMA(
    GRU,
    22,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(GRU_ver22_doc) + GenerateOptionalArgumentsDoc()))
        .Attr(
            "activations",
            "A list of 2 (or 4 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be one "
            "of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "linear_before_reset",
            "When computing the output of the hidden gate, "
            "apply the linear transformation before multiplying by the output of the "
            "reset gate.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            3,
            "B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .FillUsing(RNNDocGenerator("GRU")));

static const char* LSTM_ver22_doc = kDoc_3950deae3514;

ONNX_OPERATOR_SET_SCHEMA(
    LSTM,
    22,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(LSTM_ver22_doc) + GenerateOptionalArgumentsDoc()))
        .Attr(
            "activations",
            "A list of 3 (or 6 if bidirectional) activation functions "
            "for input, output, forget, cell, and hidden. The activation functions must "
            "be one of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "layout",
            "The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c. "
            "If 0, the following shapes are expected: "
            "X.shape = [seq_length, batch_size, input_size], "
            "Y.shape = [seq_length, num_directions, batch_size, hidden_size], "
            "initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = "
            "[num_directions, batch_size, hidden_size]. "
            "If 1, the following shapes are expected: "
            "X.shape = [batch_size, seq_length, input_size], "
            "Y.shape = [batch_size, seq_length, num_directions, hidden_size], "
            "initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = "
            "[batch_size, num_directions, hidden_size].",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("input_forget", "Couple the input and forget gates if 1.", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
            "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
            "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
            "specified - assumed to be 0.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            6,
            "initial_c",
            "Optional initial value of the cell. If not specified - assumed "
            "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            7,
            "P",
            "The weight tensor for peepholes. Concatenation of `P[iof]` and "
            "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
            "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
            "assumed to be 0.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable)
        .FillUsing(RNNDocGenerator("LSTM"))
        .Output(
            2,
            "Y_c",
            "The last output value of the cell. It has shape "
            "`[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::Differentiable));
} // namespace ONNX_NAMESPACE
