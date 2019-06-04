// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* Adam_ver11_doc = R"DOC(
    Compute one iteration of Adam, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:
     
     - The initial learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A small constant "epsilon" to avoid dividing-by-zero. 
     - Two coefficients, alpha and beta. 

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a single
    tensor "X" is being optimized. We need
    
     - the value of "X", 
     - "X"'s gradient (denoted by "G"),
     - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
     - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

    Consequently, this operator's input tensor list is ["R," "T," "X," "G," "V," "H"].
    Other parameters are given as attributes because they are usually constants.
    Moreover, the corresponding output tensors are 
    
     - the new value of "X" (called "X_new"),
     - the new exponentially-averaged historical gradient (denoted by "V_new"), and
     - the new exponentially-averaged historical squared gradient (denoted by "H_new").

    Those outputs are computed following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G;

      // Update exponentially-averaged historical gradient.
      V_new = alpha * V + (1 - alpha) * G_regularized;

      // Update exponentially-averaged historical squared gradient.
      H_new = beta * H + (1 - beta) * G_regularized * G_regularized;

      // The gradient will be element-wisely divided by the following tensor.
      H_sqrt = Sqrt(H_new) + epsilon;

      // Compute learning-rate. Note that "alpha^T"/"beta^T" is alpha's/beta's T-th power.
      R_adjusted = R * Sqrt(1 - beta^T) / (1 - alpha^T);

      // Compute new value of "X."
      X_new = X - R_adjusted * V_new / H_sqrt

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Adam,
    11,
    OpSchema()
        .SetDoc(Adam_ver11_doc)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "It sequentially contains the tensors to be optimized, the gradient, the "
            "averaged gradient (aka momentum), and the averaged squared gradient. For example, "
            "to optimize tensors \"X_1\" and \"X_2,\", the \"inputs\" would be [\"X_1\", \"X_2\", "
            "gradient of \"X_1\", gradient of \"X_2\", "
            "averaged gradient of \"X_1\", averaged gradient of \"X_2\", "
            "averaged squared gradient of \"X_1\", averaged squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "It sequentially contains the new values of optimized tensors, then the new "
            "values of averaged gradient, and finally values of averaged squared gradient. For example, "
            "if two tensors \"X_1\" and \"X_2\" are optimized, the \"outputs\" would be "
            "[new value of \"X_1,\", new value of \"X_2,\" new averaged gradient of \"X_1\", "
            "new averaged gradient of \"X_2,\" new averaged squared gradient of \"X_1,\" "
            "new averaged squared gradient of \"X_2\"].",
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
            "epsilon",
            "Small scalar to avoid dividing by zero.",
            AttributeProto::FLOAT,
            1e-6f)
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)"},
            "Constrain input types to float scalars.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Constrain output types to 64-bit integer scalars.")
        .TypeConstraint(
            "T3",
            {"tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext &ctx) {
            // Assume that the input list is [R, T, X1, X2, G1, G2, V1, V2, H1, H2] and
            // output list is [X1_new, X2_new, V1_new, V2_new, H1_new, H2_new] for explaining
            // the code below in a simpler way.

            // The count of input tensors excluding "R" and "T".
            auto num_adjustable_tensors = ctx.getNumInputs() - 2;

            // Check number of (optimized tensor, gradient, momentum) tuples.
            if (num_adjustable_tensors % 4 != 0)
              fail_shape_inference(
                  "The sum of optimized tensor count, gradient tensor count, momentum tensor count, ",
                  "accumulated squared-gradient tensor count should be a multiple of 4 in the ",
                  "\"inputs\" of Adam operator.");

            // The count of "X1" and "X2".
            auto num_optimized_tensors = num_adjustable_tensors / 4;
            for (size_t i = 0; i < num_optimized_tensors; ++i){
              // Pass X1's/X2's shapes to X1_new/X2_new.
              size_t i_in = 2 + i;
              size_t i_out = i;
              propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
              propagateShapeFromInputToOutput(ctx, i_in, i_out);

              // Pass V1's/V2's shapes to V1_new/V2_new.
              i_in = 2 + 2 * num_optimized_tensors + i;
              i_out = num_optimized_tensors + i;
              propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
              propagateShapeFromInputToOutput(ctx, i_in, i_out);

              // Pass H1's/H2's shapes to H1_new/H2_new.
              i_in = 2 + 3 * num_optimized_tensors + i;
              i_out = 2 * num_optimized_tensors + i;
              propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
              propagateShapeFromInputToOutput(ctx, i_in, i_out);
        }}));
} // namespace ONNX_NAMESPACE
