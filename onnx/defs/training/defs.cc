// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(Gradient, 1, generated::ai_onnx_preview_training::Gradient_ver1());

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adagrad,
    1,
    generated::ai_onnx_preview_training::Adagrad_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    generated::ai_onnx_preview_training::Momentum_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
    generated::ai_onnx_preview_training::Adam_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
