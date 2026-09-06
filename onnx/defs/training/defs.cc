// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Gradient,
    1,
    OpSchema()
        .SetDoc(kDoc_Gradient_ver1)
        .Input(
            0,
            "Inputs",
            "The values fed into graph identified by the attributes. "
            "The i-th input is the value of the i-th tensor specified in the "
            "concatenated list of the attribute \"xs\" and the attribute "
            " \"zs\". For example, if xs=[\"A\", \"B\"] and zs=[\"C\"], the "
            "first input is used as the value of symbol \"A\" and the 3rd "
            "input is substituted for all the occurrences of \"C\".",
            "T1",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Outputs",
            "The gradient of the tensor specified by the attribute \"y\" "
            "with respect to each of tensors specified in the "
            "attribute \"xs\". The i-th output is the gradient of \"y\" with "
            "respect to the i-th tensor specified in the attribute \"xs\".",
            "T2",
            OpSchema::Variadic,
            false)
        .Attr(
            "xs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS)
        .Attr(
            "zs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary non-differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "y",
            "The targeted tensor. It can be viewed as the output of the "
            "differentiated function. The attribute \"xs\" and attribute "
            "\"zs\" are the minimal independent variable set that determines "
            "the value of \"y\".",
            AttributeProto::STRING)
        .TypeConstraint("T1", OpSchema::all_tensor_types(), "Allow outputs to be any kind of tensor.")
        .TypeConstraint(
            "T2",
            {types::Float16, types::Float, types::Double},
            "Allow inputs to be any kind of floating-point tensor."));

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adagrad,
    1,
    OpSchema()
        .SetDoc(kDoc_Adagrad_ver1)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "The current values of optimized tensors, followed by their "
            "respective gradients, followed by their respective accumulated squared gradients."
            "For example, if two tensor \"X_1\" and \"X_2\" "
            "are optimized, "
            "The input list would be "
            "[\"X_1\", \"X_2\", "
            "gradient of \"X_1\", "
            "gradient of \"X_2\", "
            "accumulated squared gradient of \"X_1\", "
            "accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "Updated values of optimized tensors, followed by their updated "
            "values of accumulated squared gradients. For example, "
            "if two tensor \"X_1\" and \"X_2\" are "
            "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
            "new accumulated squared gradient of \"X_1\", new accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr("epsilon", "Small scalar to avoid dividing by zero.", AttributeProto::FLOAT, 1e-6f)
        .Attr(
            "decay_factor",
            "The decay factor of learning rate after one update."
            "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
            "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "norm_coefficient",
            "Regularization coefficient in 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .TypeConstraint("T1", {types::Float, types::Double}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {types::Int64}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {types::Float, types::Double}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // In comments below, we assume that the input list is
          // [R, T, X1, X2, G1, G2, H1, H2] and the output list is
          // [X1_new, X2_new, H1_new, H2_new].

          // Compute the number of tuples (X, G, H).
          auto num_optimized_tensors = (ctx.getNumInputs() - 2) / 3;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's and X2's shapes to X1_new and X2_new, respectively.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass H1's and H2's shapes to H1_new and H2_new, respectively.
            i_in = 2 + (2 * num_optimized_tensors) + i;
            i_out = i + num_optimized_tensors;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Momentum,
    1,
    OpSchema()
        .SetDoc(kDoc_Momentum_ver1)
        .Input(0, "R", "The learning rate.", "T1")
        .Input(1, "T", "Update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "It sequentially contains the current values of optimized tensors, then their "
            "gradient tensors, and finally their momentum tensors. For example, if two tensors "
            "\"X_1\" and \"X_2\" are optimized, The expected input list would be "
            "[\"X_1\", \"X_2\", gradient of \"X_1\", gradient of \"X_2\", momentum of \"X_1\", momentum of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "It sequentially contains the new values of optimized tensors and then the new "
            "values of their momentum tensors. For example, if two tensors \"X_1\" and \"X_2\" are "
            "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
            "new momentum of \"X_1\", new momentum of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr("alpha", "The decay factor of momentum. It should be a scalar.", AttributeProto::FLOAT)
        .Attr(
            "beta",
            "The coefficient of gradient in computing new momentum. It should be a scalar.",
            AttributeProto::FLOAT)
        .Attr("norm_coefficient", "Coefficient of 0.5 * norm_coefficient * ||X||^2.", AttributeProto::FLOAT)
        .Attr(
            "mode",
            "Its value should be either \"nesterov\" or \"standard\". The value \"nesterov\" leads "
            "to the use of Nesterov's momentum while \"standard\" invokes stochastic gradient method "
            "using standard momentum",
            AttributeProto::STRING)
        .TypeConstraint("T1", {types::Float, types::Double}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {types::Int64}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {types::Float, types::Double}, "Constrain input types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Assume that the input list is [R, T, X1, X2, G1, G2, V1, V2] and
          // output list is [X1_new, X2_new, V1_new, V2_new] for explaining
          // the code below in a simpler way.

          // The count of input tensors excluding "R" and "T".
          auto num_adjustable_tensors = ctx.getNumInputs() - 2;

          // Check number of (optimized tensor, gradient, momentum) tuples.
          if (num_adjustable_tensors % 3 != 0) {
            fail_shape_inference(
                "The sum of optimized tensor count and momentum tensor count ",
                "should be a multiple of 2 in the input list of Momentum operator");
          }

          // The count of "X1" and "X2".
          auto num_optimized_tensors = num_adjustable_tensors / 3;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's/X2's shapes to X1_new/X2_new.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
            // Pass V1's/V2's shapes to V1_new/V2_new.
            i_in = 2 + (2 * num_optimized_tensors) + i;
            i_out = i + num_optimized_tensors;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adam,
    1,
    OpSchema()
        .SetDoc(kDoc_Adam_ver1)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "The tensors to be optimized, followed by their respective gradients, "
            "followed by their respective accumulated gradients (aka momentum), "
            "followed by their respective accumulated squared gradients. For example, "
            "to optimize tensors \"X_1\" and \"X_2,\", the input list would be "
            "[\"X_1\", \"X_2\", "
            "gradient of \"X_1\", gradient of \"X_2\", "
            "accumulated gradient of \"X_1\", accumulated gradient of \"X_2\", "
            "accumulated squared gradient of \"X_1\", accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "New values of optimized tensors, "
            "followed by their respective new accumulated gradients, "
            "followed by their respective new accumulated squared gradients. "
            "For example, if two tensors \"X_1\" and \"X_2\" are optimized, "
            "the outputs list would be "
            "[new value of \"X_1\", new value of \"X_2\", "
            "new accumulated gradient of \"X_1\", "
            "new accumulated gradient of \"X_2\", "
            "new accumulated squared gradient of \"X_1\", "
            "new accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr(
            "alpha",
            "Coefficient of previously accumulated gradient in running average. Default to 0.9.",
            AttributeProto::FLOAT,
            0.9f)
        .Attr(
            "beta",
            "Coefficient of previously accumulated squared-gradient in running average. Default to 0.999.",
            AttributeProto::FLOAT,
            0.999f)
        .Attr(
            "norm_coefficient",
            "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "norm_coefficient_post",
            "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr("epsilon", "Small scalar to avoid dividing by zero.", AttributeProto::FLOAT, 1e-6f)
        .TypeConstraint("T1", {types::Float, types::Double}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {types::Int64}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {types::Float, types::Double}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Assume that the input list is [R, T, X1, X2, G1, G2, V1, V2, H1, H2] and
          // output list is [X1_new, X2_new, V1_new, V2_new, H1_new, H2_new] for explaining
          // the code below in a simpler way.

          // The count of input tensors excluding "R" and "T".
          auto num_adjustable_tensors = ctx.getNumInputs() - 2;

          // Check number of (optimized tensor, gradient, momentum) tuples.
          if (num_adjustable_tensors % 4 != 0) {
            fail_shape_inference(
                "The sum of optimized tensor count, gradient tensor count, momentum tensor count, ",
                "accumulated squared-gradient tensor count should be a multiple of 4 in the ",
                "\"inputs\" of Adam operator.");
          }

          // The count of "X1" and "X2".
          auto num_optimized_tensors = num_adjustable_tensors / 4;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's/X2's shapes to X1_new/X2_new.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass V1's/V2's shapes to V1_new/V2_new.
            i_in = 2 + (2 * num_optimized_tensors) + i;
            i_out = num_optimized_tensors + i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass H1's/H2's shapes to H1_new/H2_new.
            i_in = 2 + (3 * num_optimized_tensors) + i;
            i_out = (2 * num_optimized_tensors) + i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

} // namespace ONNX_NAMESPACE
