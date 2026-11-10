// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    13,
    generated::ai_onnx::GatherND_ver13().TypeAndShapeInferenceFunction(GatherNDShapeInference));
ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    18,
    generated::ai_onnx::ScatterND_ver18().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void CastDataPropagation_opset25(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset25(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    25,
    generated::ai_onnx::Cast_ver25()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset25)
        .PartialDataPropagationFunction(CastDataPropagation_opset25));

static void checked_mul_into(int64_t& accumulator, int64_t value) {
  if (checked_mul_overflow(accumulator, value, &accumulator)) {
    fail_shape_inference("Dimension product overflow in Reshape");
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    20,
    generated::ai_onnx::GridSample_ver20().TypeAndShapeInferenceFunction(gridSampleShapeInference));

static void CastDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    24,
    generated::ai_onnx::Cast_ver24()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset24)
        .PartialDataPropagationFunction(CastDataPropagation_opset24));

static void CastDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    23,
    generated::ai_onnx::Cast_ver23()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset23)
        .PartialDataPropagationFunction(CastDataPropagation_opset23));

static void CastDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    21,
    generated::ai_onnx::Cast_ver21()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset21)
        .PartialDataPropagationFunction(CastDataPropagation_opset21));

static void CastDataPropagation_opset19(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset19(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    19,
    generated::ai_onnx::Cast_ver19()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset19)
        .PartialDataPropagationFunction(CastDataPropagation_opset19));

static void CastDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void CastShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    13,
    generated::ai_onnx::Cast_ver13()
        .TypeAndShapeInferenceFunction(CastShapeInference_opset13)
        .PartialDataPropagationFunction(CastDataPropagation_opset13));

static bool CastLikeFunctionBody_opset24(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

static void CastLikeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    24,
    generated::ai_onnx::CastLike_ver24()
        .TypeAndShapeInferenceFunction(CastLikeShapeInference_opset24)
        .SetContextDependentFunctionBodyBuilder(CastLikeFunctionBody_opset24));

static bool CastLikeFunctionBody_opset23(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

static void CastLikeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    23,
    generated::ai_onnx::CastLike_ver23()
        .TypeAndShapeInferenceFunction(CastLikeShapeInference_opset23)
        .SetContextDependentFunctionBodyBuilder(CastLikeFunctionBody_opset23));

static bool CastLikeFunctionBody_opset21(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

static void CastLikeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    21,
    generated::ai_onnx::CastLike_ver21()
        .TypeAndShapeInferenceFunction(CastLikeShapeInference_opset21)
        .SetContextDependentFunctionBodyBuilder(CastLikeFunctionBody_opset21));

static bool CastLikeFunctionBody_opset19(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add(
      MakeString("output = Cast <to= ", static_cast<int64_t>(target_elt_type), ", saturate: int = @saturate> (input)")
          .c_str());
  schema.BuildFunction(functionProto);
  return true;
}

static void CastLikeShapeInference_opset19(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    19,
    generated::ai_onnx::CastLike_ver19()
        .TypeAndShapeInferenceFunction(CastLikeShapeInference_opset19)
        .SetContextDependentFunctionBodyBuilder(CastLikeFunctionBody_opset19));

static bool CastLikeFunctionBody_opset15(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto target_type = ctx.getInputType(1);
  if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
    // we cannot create a correct function body without knowing the target element type
    return false;
  }
  auto target_elt_type = target_type->tensor_type().elem_type();
  FunctionBuilder builder(functionProto);
  builder.Add("output = Cast (input)", "to", static_cast<int64_t>(target_elt_type));
  schema.BuildFunction(functionProto);
  return true;
}

static void CastLikeShapeInference_opset15(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    15,
    generated::ai_onnx::CastLike_ver15()
        .TypeAndShapeInferenceFunction(CastLikeShapeInference_opset15)
        .SetContextDependentFunctionBodyBuilder(CastLikeFunctionBody_opset15));

static void CastShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    9,
    generated::ai_onnx::Cast_ver9().TypeAndShapeInferenceFunction(CastShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    16,
    generated::ai_onnx::GridSample_ver16().TypeAndShapeInferenceFunction(gridSampleShapeInference));

static void ReshapeShapeInference_opset24(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    24,
    generated::ai_onnx::Reshape_ver24().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset24));

static void ReshapeShapeInference_opset23(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    23,
    generated::ai_onnx::Reshape_ver23().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset23));

static void ReshapeShapeInference_opset21(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    21,
    generated::ai_onnx::Reshape_ver21().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset21));

static void ReshapeShapeInference_opset19(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    19,
    generated::ai_onnx::Reshape_ver19().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset19));

static void ReshapeShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape Inference if 2nd input data (the target shape) is available
  const TensorProto* targetShapeInitializer = ctx.getInputData(1);
  if (!targetShapeInitializer) {
    return;
  }
  // Make targetShape (0 -> same as originalShape, -1 -> inferred).
  // The targetShape vector represents the specified shape for output.
  std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);
  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShape.size(), false);
  int64_t outputProduct = 1;
  for (size_t i = 0; i < targetShape.size(); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShape[i] == -1) {
      // Check if multiple -1's. If not, set negativeOneDim, marking
      // this dimension to potentially be filled in later.
      if (negativeOneDim) {
        fail_shape_inference("Target shape may not have multiple -1 dimensions");
      }
      negativeOneDim = new_dim;
    } else if (targetShape[i] == 0) {
      // Check if data input has a shape and if the index i is within
      // its bounds. If these conditions are satisfied, any dimension
      // value/param should be propagated. If dimension value cannot be
      // inferred, set the corresponding  unresolvedZeros flag to true.
      unresolvedZeros[i] = true;
      if (dataInputTensorType.has_shape()) {
        if (i >= static_cast<size_t>(dataInputTensorType.shape().dim_size())) {
          fail_shape_inference("Invalid position of 0");
        }
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
          unresolvedZeros[i] = false;
        } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
          const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
          new_dim->set_dim_param(dim_param);
        }
      }
    } else if (targetShape[i] > 0) {
      // Set the dimension value to targetShape[i]
      new_dim->set_dim_value(targetShape[i]);
      checked_mul_into(outputProduct, targetShape[i]);
    } else {
      // Check if value is less than -1; fail if so
      fail_shape_inference("Invalid dimension value: ", targetShape[i]);
    }
  }

  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    13,
    generated::ai_onnx::Reshape_ver13().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset13));

static void ReshapeShapeInference_opset5(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape Inference if 2nd input data (the target shape) is available
  const TensorProto* targetShapeInitializer = ctx.getInputData(1);
  if (!targetShapeInitializer) {
    return;
  }
  // Make targetShape (0 -> same as originalShape, -1 -> inferred).
  // The targetShape vector represents the specified shape for output.
  std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShape.size(), false);
  int64_t outputProduct = 1;
  for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShape[i] == -1) {
      // Check if multiple -1's. If not, set negativeOneDim, marking
      // this dimension to potentially be filled in later.
      if (negativeOneDim) {
        fail_shape_inference("Target shape may not have multiple -1 dimensions");
      }
      negativeOneDim = new_dim;
    } else if (targetShape[i] == 0) {
      // Check if data input has a shape and if the index i is within
      // its bounds. If these conditions are satisfied, any dimension
      // value/param should be propagated. If dimension value cannot be
      // inferred, set the corresponding  unresolvedZeros flag to true.
      unresolvedZeros[i] = true;
      if (dataInputTensorType.has_shape()) {
        if (i >= dataInputTensorType.shape().dim_size()) {
          fail_shape_inference("Invalid position of 0");
        }
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
          unresolvedZeros[i] = false;
        } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
          const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
          new_dim->set_dim_param(dim_param);
        }
      }
    } else if (targetShape[i] > 0) {
      // Set the dimension value to targetShape[i]
      new_dim->set_dim_value(targetShape[i]);
      checked_mul_into(outputProduct, targetShape[i]);
    } else {
      // Check if value is less than -1; fail if so
      fail_shape_inference("Invalid dimension value: ", targetShape[i]);
    }
  }

  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    5,
    generated::ai_onnx::Reshape_ver5().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset5));

static void ShapeDataPropagation_opset24(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

static void ShapeShapeInference_opset24(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    24,
    generated::ai_onnx::Shape_ver24()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset24)
        .PartialDataPropagationFunction(ShapeDataPropagation_opset24));

static void ShapeDataPropagation_opset23(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

static void ShapeShapeInference_opset23(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    23,
    generated::ai_onnx::Shape_ver23()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset23)
        .PartialDataPropagationFunction(ShapeDataPropagation_opset23));

// Data propagation function for Shape op
// Propagates input shape to output shape
static void ShapeOp13DataPropagator(DataPropagationContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto tsp;
    tsp.CopyFrom(input_shape);
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void ShapeShapeInference_opset13(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    13,
    generated::ai_onnx::Shape_ver13()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset13)
        .PartialDataPropagationFunction(ShapeOp13DataPropagator));

static void ShapeShapeInference_opset1(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    1,
    generated::ai_onnx::Shape_ver1()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset1)
        .PartialDataPropagationFunction(ShapeOp13DataPropagator));

static void SizeDataPropagation_opset24(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void SizeShapeInference_opset24(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    24,
    generated::ai_onnx::Size_ver24()
        .TypeAndShapeInferenceFunction(SizeShapeInference_opset24)
        .PartialDataPropagationFunction(SizeDataPropagation_opset24));

static void SizeDataPropagation_opset23(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void SizeShapeInference_opset23(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    23,
    generated::ai_onnx::Size_ver23()
        .TypeAndShapeInferenceFunction(SizeShapeInference_opset23)
        .PartialDataPropagationFunction(SizeDataPropagation_opset23));

static void SizeShapeInference_opset1(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    1,
    generated::ai_onnx::Size_ver1().TypeAndShapeInferenceFunction(SizeShapeInference_opset1));

static void ConcatShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
    return;
  }

  auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis must be in [-rank, rank-1].");
  }
  if (axis < 0) {
    axis += rank;
  }

  if (numInputs == 1) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
    return;
  }

  bool all_lengths_known = true;
  int64_t total_length = 0;

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  for (size_t i = 0; i < numInputs; i++) {
    const auto& shape = ctx.getInputType(i)->tensor_type().shape();
    if (shape.dim_size() != rank) {
      fail_shape_inference(
          "All inputs to Concat must have same rank. Input ", i, " has rank ", shape.dim_size(), " != ", rank);
    }
    for (int j = 0; j < rank; j++) {
      if (j == axis) {
        if (shape.dim(j).has_dim_value()) {
          const int64_t dim_val = shape.dim(j).dim_value();
          if (dim_val < 0) {
            fail_shape_inference("Negative dimension value on Concat axis");
          }
          if (checked_add_overflow(total_length, dim_val, &total_length)) {
            fail_shape_inference("Integer overflow computing Concat output length");
          }
        } else {
          all_lengths_known = false;
        }
      } else {
        auto& output_dim = *output_shape->mutable_dim(j);
        const auto& input_dim = shape.dim(j);
        mergeInDimensionInfo(input_dim, output_dim, j);
      }
    }
  }

  if (all_lengths_known) {
    output_shape->mutable_dim(axis)->set_dim_value(total_length);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    11,
    generated::ai_onnx::Concat_ver11().TypeAndShapeInferenceFunction(ConcatShapeInference_opset11));

static void SplitShapeInference_opset11(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  if (getRepeatedAttribute(ctx, "split", split)) {
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else {
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    11,
    generated::ai_onnx::Split_ver11().TypeAndShapeInferenceFunction(SplitShapeInference_opset11));

static void SplitShapeInference_opset13(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  size_t num_inputs = ctx.getNumInputs();
  if ((num_inputs == 2) && ctx.getInputType(1)) { //'split' is input
    auto split_proto = ctx.getInputData(1);
    if (split_proto == nullptr) {
      // skip if split is not an initializer
      return;
    }
    split = ParseData<int64_t>(split_proto);
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else { // no value available for 'split'
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    split.reserve(ctx.getNumOutputs());
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    13,
    generated::ai_onnx::Split_ver13().TypeAndShapeInferenceFunction(SplitShapeInference_opset13));

static void SliceShapeInference_opset11(InferenceContext& ctx) {
  size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
    fail_type_inference("Slice op must have either three, four or five inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  // Shape Inference if
  //     1. 2nd and 3rd input data (starts, ends) are available.
  // and 2. 4th and 5th optional input (axes, steps) are either not set,
  // or set and is initializer.
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    // we can infer the output rank - it never changes
    for (int i = 0; i < input_rank; ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  // don't know data_type- can't proceed
  if (!startsInitializer->has_data_type())
    return;

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }

  std::vector<int64_t> steps;
  if (!stepsInitializer) {
    steps = std::vector<int64_t>(starts.size(), 1);
  } else {
    steps = get_initializer_data(stepsInitializer);
    if (steps.size() != axes.size()) {
      fail_shape_inference("Input steps has incorrect length");
    }
  }

  for (int i = 0; i < input_rank; ++i) {
    // first update rank of output dim
    auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      output_dim->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      output_dim->set_dim_param(input_dim.dim_param());
    }
  }

  std::unordered_set<int64_t> unique_axes;
  size_t axes_size = axes.size();
  for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

    if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
      fail_shape_inference("Input axes has invalid data");
    }

    if (unique_axes.find(axis) != unique_axes.end()) {
      fail_shape_inference("'axes' has duplicates");
    }

    unique_axes.insert(axis);

    auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

    // input dim value is missing - cannot perform shape inference for
    // this axis
    if (!input_dim.has_dim_value()) {
      // Clear any previously propagated dim_param and leave this
      // dimension "empty", before moving on to the next dimension
      ctx.getOutputType(0)
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(static_cast<int>(axis))
          ->clear_dim_param();
      continue;
    }

    const auto input_dim_value = input_dim.dim_value();

    // A negative dimension is invalid: the std::clamp calls below require the
    // upper bound to be >= the lower bound, so reject the dimension with a
    // catchable error instead of invoking undefined behavior (a hardened
    // std::clamp aborts the process with SIGABRT).
    if (input_dim_value < 0) {
      fail_shape_inference("Slice input dimension must be non-negative, got ", input_dim_value);
    }

    // Empty dimension: clamp bounds are invalid when dimension size is 0,
    // so short-circuit to produce a zero-length output.
    if (input_dim_value == 0) {
      ctx.getOutputType(0)
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(static_cast<int>(axis))
          ->set_dim_value(0);
      continue;
    }

    // process step
    auto step = steps[axis_index];
    if (step == 0) {
      fail_shape_inference("'step' cannot be 0");
    }

    // process start
    auto start = starts[axis_index];
    if (start < 0)
      start += input_dim_value;
    if (step < 0)
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
    else
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

    // process end
    auto end = ends[axis_index];
    if (end < static_cast<int64_t>(0))
      end += input_dim_value;
    if (step < static_cast<int64_t>(0))
      end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
    else
      end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
    if (temp < 0)
      temp = 0;

    // assign output value
    ctx.getOutputType(0)
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(static_cast<int>(axis))
        ->set_dim_value(temp);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    11,
    generated::ai_onnx::Slice_ver11().TypeAndShapeInferenceFunction(SliceShapeInference_opset11));

static void TransposeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    24,
    generated::ai_onnx::Transpose_ver24().TypeAndShapeInferenceFunction(TransposeShapeInference_opset24));

static void TransposeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    23,
    generated::ai_onnx::Transpose_ver23().TypeAndShapeInferenceFunction(TransposeShapeInference_opset23));

static void TransposeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    21,
    generated::ai_onnx::Transpose_ver21().TypeAndShapeInferenceFunction(TransposeShapeInference_opset21));

static void TransposeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    perm.reserve(shape.dim_size());
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  getOutputShape(ctx, 0);

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    13,
    generated::ai_onnx::Transpose_ver13().TypeAndShapeInferenceFunction(TransposeShapeInference_opset13));

static void TransposeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  auto input_type = ctx.getInputType(0);
  const TensorShapeProto& shape = input_type->tensor_type().shape();
  std::vector<int64_t> perm;
  bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
  if (!has_perm_attr) {
    for (int i = shape.dim_size() - 1; i >= 0; --i)
      perm.push_back(i);
  } else if (!perm.empty()) {
    // check if every index is valid
    std::vector<bool> seen(shape.dim_size(), false);
    for (int64_t fromDimIndex : perm) {
      if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
        std::ostringstream oss;
        oss << "Invalid attribute perm {" << perm[0];
        for (size_t i = 1; i != perm.size(); ++i) {
          oss << ", " << perm[i];
        }
        oss << "}, input shape = {";
        if (shape.dim_size() > 0) {
          oss << shape.dim(0).dim_value();
          for (int i = 1; i != shape.dim_size(); ++i) {
            oss << ", " << shape.dim(i).dim_value();
          }
          oss << "}";
        }
        fail_type_inference(oss.str());
      } else {
        // check if any perm is repeated
        if (seen[fromDimIndex]) {
          fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
        }
        seen[fromDimIndex] = true;
      }
    }
  }

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  for (int64_t i : perm) {
    appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    1,
    generated::ai_onnx::Transpose_ver1().TypeAndShapeInferenceFunction(TransposeShapeInference_opset1));

static void ScatterNDShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    16,
    generated::ai_onnx::ScatterND_ver16().TypeAndShapeInferenceFunction(ScatterNDShapeInference_opset16));

static void ScatterNDShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    13,
    generated::ai_onnx::ScatterND_ver13().TypeAndShapeInferenceFunction(ScatterNDShapeInference_opset13));

static void ScatterNDShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    11,
    generated::ai_onnx::ScatterND_ver11().TypeAndShapeInferenceFunction(ScatterNDShapeInference_opset11));

static void ScatterElementsShapeInference_opset16(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    16,
    generated::ai_onnx::ScatterElements_ver16().TypeAndShapeInferenceFunction(ScatterElementsShapeInference_opset16));

static void ScatterElementsShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    13,
    generated::ai_onnx::ScatterElements_ver13().TypeAndShapeInferenceFunction(ScatterElementsShapeInference_opset13));

static void ScatterElementsShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    11,
    generated::ai_onnx::ScatterElements_ver11().TypeAndShapeInferenceFunction(ScatterElementsShapeInference_opset11));

static void GatherShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  int r = data_shape.dim_size();
  if (r < 1) {
    fail_shape_inference("data tensor must have rank >= 1");
  }
  int q = indices_shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis >= r) {
    fail_shape_inference("axis must be in [-r, r-1]");
  }
  if (axis < 0) {
    axis += r;
  }
  int out_rank = q + r - 1;
  if (out_rank == 0) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
  for (int i = 0; i < out_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                          : // i < axis < r
        (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                    : // i - axis < q
        data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    11,
    generated::ai_onnx::Gather_ver11()
        .TypeAndShapeInferenceFunction(GatherShapeInference_opset11)
        .PartialDataPropagationFunction(GatherOp13DataPropagator));

static void GatherElementsShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // propagate indices' shape to output if it exists
  if (hasInputShape(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 1, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    11,
    generated::ai_onnx::GatherElements_ver11().TypeAndShapeInferenceFunction(GatherElementsShapeInference_opset11));

static void SqueezeDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void SqueezeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    24,
    generated::ai_onnx::Squeeze_ver24()
        .TypeAndShapeInferenceFunction(SqueezeShapeInference_opset24)
        .PartialDataPropagationFunction(SqueezeDataPropagation_opset24));

static void SqueezeDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void SqueezeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    23,
    generated::ai_onnx::Squeeze_ver23()
        .TypeAndShapeInferenceFunction(SqueezeShapeInference_opset23)
        .PartialDataPropagationFunction(SqueezeDataPropagation_opset23));

static void SqueezeDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void SqueezeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    21,
    generated::ai_onnx::Squeeze_ver21()
        .TypeAndShapeInferenceFunction(SqueezeShapeInference_opset21)
        .PartialDataPropagationFunction(SqueezeDataPropagation_opset21));

static void SqueezeDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void SqueezeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  size_t num_inputs = ctx.getNumInputs();
  bool axes_not_specified = false;

  if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
    auto axes_proto = ctx.getInputData(1);
    if (axes_proto == nullptr) {
      // skip if axes is not an initializer
      return;
    }
    axes = ParseData<int64_t>(axes_proto);
  } else {
    // axes not specified
    axes_not_specified = true;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  checkAxesRange(axes, input_ndim);
  adjustNegativeAxes(axes, input_ndim);

  for (int i = 0; i < input_ndim; ++i) {
    if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
      // if dim has a symbolic value and the axes spec want to act on all dims,
      // return early because we can't infer the shape
      return;
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int i = 0; i < input_ndim; ++i) {
    if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
      // if axes not specified, do not keep shape if the dimension is equal to one
      continue;
    } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
      // if axes wants to explicitly act on this dim, fail explicitly only if the
      // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
      // the dim is symbolic, runtime engines should check that the dimension is
      // actually 1 when the op is evaluated
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    13,
    generated::ai_onnx::Squeeze_ver13()
        .TypeAndShapeInferenceFunction(SqueezeShapeInference_opset13)
        .PartialDataPropagationFunction(SqueezeDataPropagation_opset13));

static void SqueezeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (int i = 0; i < input_ndim; ++i) {
      if (!input_shape.dim(i).has_dim_value()) {
        return;
      }
      if (input_shape.dim(i).dim_value() == 1) {
        axes.push_back(i);
      }
    }
  }

  std::transform(axes.begin(), axes.end(), axes.begin(), [&](int64_t axis) -> int64_t {
    return axis < 0 ? axis + input_ndim : axis;
  });

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < input_ndim; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    11,
    generated::ai_onnx::Squeeze_ver11().TypeAndShapeInferenceFunction(SqueezeShapeInference_opset11));

static void UnsqueezeDataPropagation_opset24(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void UnsqueezeShapeInference_opset24(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    24,
    generated::ai_onnx::Unsqueeze_ver24()
        .TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset24)
        .PartialDataPropagationFunction(UnsqueezeDataPropagation_opset24));

static void UnsqueezeDataPropagation_opset23(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void UnsqueezeShapeInference_opset23(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    23,
    generated::ai_onnx::Unsqueeze_ver23()
        .TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset23)
        .PartialDataPropagationFunction(UnsqueezeDataPropagation_opset23));

static void UnsqueezeDataPropagation_opset21(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void UnsqueezeShapeInference_opset21(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    21,
    generated::ai_onnx::Unsqueeze_ver21()
        .TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset21)
        .PartialDataPropagationFunction(UnsqueezeDataPropagation_opset21));

static void UnsqueezeDataPropagation_opset13(DataPropagationContext& ctx) {
  PropagateShapeDataFromInputToOutput(ctx, 0);
}

static void UnsqueezeShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> axes;
  auto axes_proto = ctx.getInputData(1);
  if (axes_proto == nullptr) {
    // skip if axes is not an initializer
    return;
  }
  axes = ParseData<int64_t>(axes_proto);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  checkAxesRange(axes, output_ndim);
  adjustNegativeAxes(axes, output_ndim);
  checkDuplicateAxes(axes, output_ndim);
  // sort after correcting negative axes values (if any)
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    13,
    generated::ai_onnx::Unsqueeze_ver13()
        .TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset13)
        .PartialDataPropagationFunction(UnsqueezeDataPropagation_opset13));

static void UnsqueezeShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    return;
  }

  // validate 'axes' for duplicate entries
  std::unordered_set<int64_t> unique_values;
  for (const auto val : axes) {
    if (unique_values.find(val) != unique_values.end()) {
      fail_shape_inference("'axes' attribute must not contain any duplicates");
    }
    unique_values.insert(val);
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  const auto output_ndim = input_ndim + static_cast<int>(axes.size());
  for (auto& axe : axes) {
    if (axe < -output_ndim || axe >= output_ndim) {
      fail_shape_inference("values in 'axes' are beyond the bounds of the computed output shape");
    }
    if (axe < 0) {
      axe += output_ndim;
    }
  }

  // sort after correcting negative axes values (if any) in the previous
  // step
  std::sort(axes.begin(), axes.end());

  int j = 0;
  for (int i = 0; i < input_ndim; ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    11,
    generated::ai_onnx::Unsqueeze_ver11().TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset11));

static void SpaceToDepthShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) * block_area,
           input_shape.dim(2) / blocksize,
           input_shape.dim(3) / blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    1,
    generated::ai_onnx::SpaceToDepth_ver1().TypeAndShapeInferenceFunction(SpaceToDepthShapeInference_opset1));

static void SpaceToDepthShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) * block_area,
           input_shape.dim(2) / blocksize,
           input_shape.dim(3) / blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    13,
    generated::ai_onnx::SpaceToDepth_ver13().TypeAndShapeInferenceFunction(SpaceToDepthShapeInference_opset13));

static void DepthToSpaceShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    11,
    generated::ai_onnx::DepthToSpace_ver11().TypeAndShapeInferenceFunction(DepthToSpaceShapeInference_opset11));

static void DepthToSpaceShapeInference_opset13(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    13,
    generated::ai_onnx::DepthToSpace_ver13().TypeAndShapeInferenceFunction(DepthToSpaceShapeInference_opset13));

static void TileShapeInference_opset6(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference

  // Needs at least the first input to proceed
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  const auto repeats_inputs = ctx.getInputData(1);

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
    // shape inference is possible only when 'repeats' is an initializer
    const auto& repeats_shape = ctx.getInputType(1)->tensor_type().shape();
    if (repeats_shape.dim_size() != 1 || repeats_inputs->data_type() != TensorProto::INT64) {
      fail_shape_inference("'Repeats' input must be 1D tensor of type int64");
    }

    const auto repeats_data = ParseData<int64_t>(repeats_inputs);

    if (repeats_data.size() != static_cast<size_t>(input_rank)) {
      fail_shape_inference(
          "'Repeats' input has incorrect number of values. "
          "The number of values in 'repeats' must be equal "
          "to the number of input dimensions.");
    }

    for (int i = 0; i < input_rank; ++i) {
      if (repeats_data[i] < 0) {
        fail_shape_inference("'Repeats' input must not contain negative values");
      }
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedMultiply(input_dim.dim_value(), repeats_data[i]));
      }
    }
  } else {
    // Infer output shape's rank in any case (if repeats data is not
    // available)
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    6,
    generated::ai_onnx::Tile_ver6().TypeAndShapeInferenceFunction(TileShapeInference_opset6));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    18,
    generated::ai_onnx::Resize_ver18().TypeAndShapeInferenceFunction(resizeShapeInference_opset13_to_18));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    13,
    generated::ai_onnx::Resize_ver13().TypeAndShapeInferenceFunction(resizeShapeInference_opset13_to_18));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    11,
    generated::ai_onnx::Resize_ver11().TypeAndShapeInferenceFunction(resizeShapeInference_opset11_to_12));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    24,
    generated::ai_onnx::Identity_ver24().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    23,
    generated::ai_onnx::Identity_ver23().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    21,
    generated::ai_onnx::Identity_ver21().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    19,
    generated::ai_onnx::Identity_ver19().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    13,
    generated::ai_onnx::Identity_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    1,
    generated::ai_onnx::Identity_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void IsNaNShapeInference_opset9(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    9,
    generated::ai_onnx::IsNaN_ver9().TypeAndShapeInferenceFunction(IsNaNShapeInference_opset9));

static void IsNaNShapeInference_opset13(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    13,
    generated::ai_onnx::IsNaN_ver13().TypeAndShapeInferenceFunction(IsNaNShapeInference_opset13));

static void IsInfShapeInference_opset10(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::BOOL);
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    10,
    generated::ai_onnx::IsInf_ver10().TypeAndShapeInferenceFunction(IsInfShapeInference_opset10));

static void NonZeroShapeInference_opset9(InferenceContext& ctx) {
  updateOutputElemType(ctx, 0, TensorProto::INT64);
  TensorShapeProto output_shape;
  auto dim = output_shape.add_dim();
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& input_shape = getInputShape(ctx, 0);
    dim->set_dim_value(input_shape.dim_size());
  }
  output_shape.add_dim();
  updateOutputShape(ctx, 0, output_shape);
}

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    9,
    generated::ai_onnx::NonZero_ver9().TypeAndShapeInferenceFunction(NonZeroShapeInference_opset9));

static void GatherNDShapeInference_opset12(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (!hasNInputShapes(ctx, 2)) {
    // cannot proceed with shape or rank inference
    return;
  }

  const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto data_rank = data_shape.dim_size();

  const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  const auto indices_rank = indices_shape.dim_size();

  int64_t batch_dims_data = getAttribute(ctx, "batch_dims", 0);
  if (data_rank < 1 || indices_rank < 1) {
    fail_shape_inference(
        "Both `data` and `indices` input tensors in GatherND op "
        "need to have rank larger than 0.");
  }
  if (batch_dims_data < 0) {
    fail_shape_inference("attribute 'batch_dims' of 'GatherND' must not be negative");
  }

  // cannot ascertain if the input shapes are valid if shape of
  // `indices` is missing last dimension value so return at this point
  if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
    return;
  }

  const auto indices_last_dim = indices_shape.dim(indices_rank - 1).dim_value();
  if (indices_last_dim < 0) {
    fail_shape_inference("Last dimension of `indices` input tensor in GatherND op must not be negative");
  }
  const auto last_index_dimension = checkedAdd(indices_last_dim, batch_dims_data);

  if (last_index_dimension > data_rank) {
    fail_shape_inference(
        "Last dimension of `indices` input tensor in GatherND op "
        "must not be larger than the rank of `data` tensor");
  }

  for (int i = 0; i < indices_rank - 1; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
  }

  for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    12,
    generated::ai_onnx::GatherND_ver12().TypeAndShapeInferenceFunction(GatherNDShapeInference_opset12));

ONNX_OPERATOR_SET_SCHEMA(Pad, 24, generated::ai_onnx::Pad_ver24().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 23, generated::ai_onnx::Pad_ver23().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 21, generated::ai_onnx::Pad_ver21().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Pad, 19, generated::ai_onnx::Pad_ver19().FillUsing(PadDocGenerator()));

static void PadShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  // Infer output shape if 'pads' tensor is available
  const auto pads_initializer = ctx.getInputData(1);
  if (nullptr != pads_initializer) {
    if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
      fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
    }

    const auto pads_data = ParseData<int64_t>(pads_initializer);
    if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
      fail_shape_inference("Pads has incorrect number of values");
    }

    auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < input_rank; ++i) {
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
      } else if (total_pad == 0) {
        *output_dim = input_dim;
      }
    }
  } else {
    // Infer output shapes' rank in any case
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    11,
    generated::ai_onnx::Pad_ver11().TypeAndShapeInferenceFunction(PadShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(Cast, 1, generated::ai_onnx::Cast_ver1());

static void CastShapeInference_opset6(InferenceContext& ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    6,
    generated::ai_onnx::Cast_ver6().TypeAndShapeInferenceFunction(CastShapeInference_opset6));

ONNX_OPERATOR_SET_SCHEMA(Concat, 1, generated::ai_onnx::Concat_ver1());

static void ConcatShapeInference_opset4(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
    return;
  }

  auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    fail_shape_inference("Required attribute axis is missing");
  }
  int axis = static_cast<int>(axisAttr->i());
  if (rank <= axis) {
    fail_shape_inference("rank must be greater than axis");
  }
  if (axis < 0) {
    return;
  }

  bool all_lengths_known = true;
  int64_t total_length = 0;

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int64_t i = 0; i < rank; ++i) {
    output_shape->add_dim();
  }

  for (size_t i = 0; i < numInputs; i++) {
    const auto& shape = ctx.getInputType(i)->tensor_type().shape();
    if (shape.dim_size() != rank) {
      fail_shape_inference("All inputs to Concat must have same rank");
    }
    for (int j = 0; j < rank; j++) {
      if (j == axis) {
        if (shape.dim(j).has_dim_value()) {
          const int64_t dim_val = shape.dim(j).dim_value();
          if (dim_val < 0) {
            fail_shape_inference("Negative dimension value on Concat axis");
          }
          if (checked_add_overflow(total_length, dim_val, &total_length)) {
            fail_shape_inference("Integer overflow computing Concat output length");
          }
        } else {
          all_lengths_known = false;
        }
      } else {
        auto& output_dim = *output_shape->mutable_dim(j);
        const auto& input_dim = shape.dim(j);
        mergeInDimensionInfo(input_dim, output_dim, j);
      }
    }
  }

  if (all_lengths_known) {
    output_shape->mutable_dim(axis)->set_dim_value(total_length);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    4,
    generated::ai_onnx::Concat_ver4().TypeAndShapeInferenceFunction(ConcatShapeInference_opset4));

ONNX_OPERATOR_SET_SCHEMA(Split, 1, generated::ai_onnx::Split_ver1());

ONNX_OPERATOR_SET_SCHEMA(Pad, 1, generated::ai_onnx::Pad_ver1());

ONNX_OPERATOR_SET_SCHEMA(Reshape, 1, generated::ai_onnx::Reshape_ver1());

static void TileShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Only rank of output can be inferred. We can do better if second
  // input is a constant, but this requires extending InferenceContext
  // interface to get values of constant inputs.
}

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    1,
    generated::ai_onnx::Tile_ver1().TypeAndShapeInferenceFunction(TileShapeInference_opset1));

ONNX_OPERATOR_SET_SCHEMA(Upsample, 1, generated::ai_onnx::Upsample_ver1());

static void UpsampleShapeInference_opset7(InferenceContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const auto& input_shape = getInputShape(ctx, 0);
  auto output_shape = getOutputShape(ctx, 0);
  const auto scales = ctx.getAttribute("scales");

  if (output_shape->dim_size() > 0) {
    if (output_shape->dim_size() != input_shape.dim_size()) {
      fail_shape_inference(
          "Ranks inferred (",
          input_shape.dim_size(),
          ") is not equal to the existing rank value (",
          output_shape->dim_size(),
          ").");
    }
  } else { // Infer the rank of output anyway
    for (int i = 0; i < input_shape.dim_size(); ++i) {
      output_shape->add_dim();
    }
  }

  if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (scales->type() == AttributeProto_AttributeType_FLOATS) {
      const std::vector<float> scales_data(scales->floats().begin(), scales->floats().end());
      if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
        fail_shape_inference("Number of elements of attribute 'scales' must be same as rank of input 'X'");
      }
      resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
    } else {
      fail_shape_inference("Attribute 'scales' must have floats type.");
    } // scales->type() == float
  } else {
    fail_shape_inference("Attribute 'scales' is required.");
  } // nullptr != scales
}

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    7,
    generated::ai_onnx::Upsample_ver7().TypeAndShapeInferenceFunction(UpsampleShapeInference_opset7));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    9,
    generated::ai_onnx::Upsample_ver9().TypeAndShapeInferenceFunction(resizeShapeInference_opset7_to_10));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    10,
    generated::ai_onnx::Resize_ver10().TypeAndShapeInferenceFunction(resizeShapeInference_opset7_to_10));

static void SliceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  if (!getRepeatedAttribute(ctx, "starts", starts) || !getRepeatedAttribute(ctx, "ends", ends) ||
      starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing attribute value for starts and ends");
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(i);
    }
  } else if (axes.size() != starts.size()) {
    fail_shape_inference("Attribute axes has incorrect length");
  } else if (!std::is_sorted(axes.begin(), axes.end())) {
    // TODO(ONNX) support shape inference for unsorted axes
    return;
  }

  auto is_negative = [](int64_t index) { return index < 0; };
  if (std::any_of(axes.begin(), axes.end(), is_negative)) {
    // Negative axes were not explicitly discussed in the spec before opset-10.
    // Hence, they are officially not part of the spec, but some models/runtimes may use them.
    // So we perform simple rank inference in this case.
    for (size_t i = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (size_t i = 0, j = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
    auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    if (j < axes.size() && static_cast<size_t>(axes[j]) == i) {
      // There's a lot of potential behaviors. For now just
      // handle some simple cases.
      const auto& dim = ctx.getInputType(0)->tensor_type().shape().dim(i);
      if (dim.has_dim_value()) {
        auto dim_value = dim.dim_value();
        if (starts[j] < 0) {
          starts[j] += dim_value;
        }
        if (ends[j] < 0) {
          ends[j] += dim_value;
        }
        if (starts[j] >= 0 && ends[j] >= 0) {
          auto newval = std::min(dim_value, ends[j]) - starts[j];
          if (newval >= 0) {
            newdim->set_dim_value(newval);
          }
        }
      }
      ++j;
    } else {
      *newdim = ctx.getInputType(0)->tensor_type().shape().dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    1,
    generated::ai_onnx::Slice_ver1().TypeAndShapeInferenceFunction(SliceShapeInference_opset1));

static void SliceShapeInference_opset10(InferenceContext& ctx) {
  size_t num_inputs = ctx.getNumInputs();
  if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
    fail_type_inference("Slice op must have either three, four or five inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  // Shape Inference if
  //     1. 2nd and 3rd input data (starts, ends) are available.
  // and 2. 4th and 5th optional input (axes, steps) are either not set,
  // or set and is initializer.
  const TensorProto* startsInitializer = ctx.getInputData(1);
  const TensorProto* endsInitializer = ctx.getInputData(2);
  const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
  const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

  if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
      (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
    const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    // we can infer the output rank - it never changes
    for (int i = 0; i < input_rank; ++i) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    }
    return;
  }

  // don't know data_type- can't proceed
  if (!startsInitializer->has_data_type())
    return;

  auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
    std::vector<int64_t> vec;
    if (initializer->data_type() == TensorProto::INT64) {
      const auto data = ParseData<int64_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else if (initializer->data_type() == TensorProto::INT32) {
      const auto data = ParseData<int32_t>(initializer);
      vec.insert(vec.end(), data.begin(), data.end());
    } else {
      // unaccepted data type
      fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
    }
    return vec;
  };

  std::vector<int64_t> starts = get_initializer_data(startsInitializer);
  std::vector<int64_t> ends = get_initializer_data(endsInitializer);

  if (starts.size() != ends.size()) {
    fail_shape_inference("Incorrect or missing input value for starts and ends");
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();
  std::vector<int64_t> axes(starts.size());
  if (!axesInitializer) {
    std::iota(axes.begin(), axes.end(), 0);
  } else {
    axes = get_initializer_data(axesInitializer);
    if (axes.size() != starts.size()) {
      fail_shape_inference("Input axes has incorrect length");
    }
  }

  std::vector<int64_t> steps;
  if (!stepsInitializer) {
    steps = std::vector<int64_t>(starts.size(), 1);
  } else {
    steps = get_initializer_data(stepsInitializer);
    if (steps.size() != axes.size()) {
      fail_shape_inference("Input steps has incorrect length");
    }
  }

  for (int i = 0; i < input_rank; ++i) {
    // first update rank of output dim
    auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto& input_dim = input_shape.dim(i);
    if (input_dim.has_dim_value()) {
      output_dim->set_dim_value(input_dim.dim_value());
    } else if (input_dim.has_dim_param()) {
      output_dim->set_dim_param(input_dim.dim_param());
    }
  }

  std::unordered_set<int64_t> unique_axes;
  size_t axes_size = axes.size();
  for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
    auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

    if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
      fail_shape_inference("Input axes has invalid data");
    }

    if (unique_axes.find(axis) != unique_axes.end()) {
      fail_shape_inference("'axes' has duplicates");
    }

    unique_axes.insert(axis);
    auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(axis);

    // input dim value is missing - cannot perform shape inference for
    // this axis
    if (!input_dim.has_dim_value())
      continue;

    const auto input_dim_value = input_dim.dim_value();

    // A negative dimension is invalid: the std::clamp calls below require the
    // upper bound to be >= the lower bound, so reject the dimension with a
    // catchable error instead of invoking undefined behavior (a hardened
    // std::clamp aborts the process with SIGABRT).
    if (input_dim_value < 0) {
      fail_shape_inference("Slice input dimension must be non-negative, got ", input_dim_value);
    }

    // Empty dimension: clamp bounds are invalid when dimension size is 0,
    // so short-circuit to produce a zero-length output.
    if (input_dim_value == 0) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(0);
      continue;
    }

    // process step
    auto step = steps[axis_index];
    if (step == 0) {
      fail_shape_inference("'step' cannot be 0");
    }

    // process start
    auto start = starts[axis_index];
    if (start < 0)
      start += input_dim_value;
    if (step < 0)
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
    else
      start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

    // process end
    auto end = ends[axis_index];
    if (end < static_cast<int64_t>(0))
      end += input_dim_value;
    if (step < static_cast<int64_t>(0))
      end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
    else
      end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

    // find output dim value for this axis
    auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
    if (temp < 0)
      temp = 0;

    // assign output value
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(temp);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    10,
    generated::ai_onnx::Slice_ver10().TypeAndShapeInferenceFunction(SliceShapeInference_opset10));

static void ScatterShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (hasNInputShapes(ctx, 1)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    9,
    generated::ai_onnx::Scatter_ver9().TypeAndShapeInferenceFunction(ScatterShapeInference_opset9));

static void DepthToSpaceShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }
  const auto block_area = checkedMultiply(blocksize, blocksize);
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(
          ctx,
          0,
          {input_shape.dim(0),
           input_shape.dim(1) / block_area,
           input_shape.dim(2) * blocksize,
           input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    1,
    generated::ai_onnx::DepthToSpace_ver1().TypeAndShapeInferenceFunction(DepthToSpaceShapeInference_opset1));

static void GatherShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  int r = data_shape.dim_size();
  if (r < 1) {
    fail_shape_inference("data tensor must have rank >= 1");
  }
  int q = indices_shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -r || axis >= r) {
    fail_shape_inference("axis must be in [-r, r-1]");
  }
  if (axis < 0) {
    axis += r;
  }
  int out_rank = q + r - 1;

  if (out_rank == 0) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  }
  for (int i = 0; i < out_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                          : // i < axis < r
        (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                    : // i - axis < q
        data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    1,
    generated::ai_onnx::Gather_ver1()
        .TypeAndShapeInferenceFunction(GatherShapeInference_opset1)
        .PartialDataPropagationFunction(GatherOp13DataPropagator));

static void SqueezeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_ndim = input_shape.dim_size();
  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    for (int i = 0; i < input_ndim; ++i) {
      if (!input_shape.dim(i).has_dim_value()) {
        return;
      }
      if (input_shape.dim(i).dim_value() == 1) {
        axes.push_back(i);
      }
    }
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0, j = 0; i < input_shape.dim_size(); ++i) {
    if (static_cast<size_t>(j) < axes.size() && axes[j] == i) {
      if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
        fail_shape_inference("Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
      }
      ++j;
    } else {
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    1,
    generated::ai_onnx::Squeeze_ver1().TypeAndShapeInferenceFunction(SqueezeShapeInference_opset1));

static void UnsqueezeShapeInference_opset1(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  std::vector<int64_t> axes;
  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    return;
  }
  std::sort(axes.begin(), axes.end());

  if (!ctx.getInputType(0)->tensor_type().has_shape()) {
    return;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  int j = 0;
  for (int i = 0; i < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
    while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      ++j;
    }
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
        ctx.getInputType(0)->tensor_type().shape().dim(i);
  }
  while (static_cast<size_t>(j) < axes.size() && axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
    ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    ++j;
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    1,
    generated::ai_onnx::Unsqueeze_ver1().TypeAndShapeInferenceFunction(UnsqueezeShapeInference_opset1));

static void OneHotShapeInference_opset9(InferenceContext& ctx) {
  oneHotShapeInference(ctx, 9);
}

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    9,
    generated::ai_onnx::OneHot_ver9().TypeAndShapeInferenceFunction(OneHotShapeInference_opset9));

ONNX_OPERATOR_SET_SCHEMA(Compress, 9, generated::ai_onnx::Compress_ver9());

static void SplitShapeInference_opset2(InferenceContext& ctx) {
  if (ctx.getNumOutputs() == 0) {
    fail_shape_inference("Split must have at least one output");
  }
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0, i);
  }
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  const auto& shape = ctx.getInputType(0)->tensor_type().shape();
  int rank = shape.dim_size();
  int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
  if (axis < -rank || axis >= rank) {
    fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
  }
  // Previously Split-2 does not mention how to deal with negative axis
  // However, there is an existing test onnx/backend/test/data/pytorch-converted/test_GLU
  // using Split-2 with negative axis and it is hard to be regenerated.
  // To compromise, handle negative axis for Split-2 here.
  if (axis < 0) {
    axis += rank;
  }
  const auto& split_dim = shape.dim(axis);
  if (!split_dim.has_dim_value()) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
      ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
    }
    return;
  }
  int split_dim_value = static_cast<int>(split_dim.dim_value());

  std::vector<int64_t> split;
  if (getRepeatedAttribute(ctx, "split", split)) {
    if (split.size() != ctx.getNumOutputs()) {
      fail_shape_inference(
          "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
    }
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim_value) {
      fail_shape_inference(
          "Mismatch between the sum of 'split' (",
          total_dim,
          ") and the split dimension of the input (",
          split_dim_value,
          ")");
    }
  } else {
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim_value % num_outputs != 0) {
      fail_shape_inference("The input is not evenly splittable");
    }
    int chunk_size = split_dim_value / num_outputs;
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
    ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    2,
    generated::ai_onnx::Split_ver2().TypeAndShapeInferenceFunction(SplitShapeInference_opset2));

static void PadShapeInference_opset2(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto& input_shape = ctx.getInputType(0)->tensor_type().shape();

  std::vector<int64_t> pads;
  if (!getRepeatedAttribute(ctx, "pads", pads)) {
    fail_shape_inference("Attribute value for pads is required");
  }
  if (pads.size() != static_cast<size_t>(input_shape.dim_size() * 2)) {
    fail_shape_inference("Attribute pads has incorrect length");
    ;
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
    const auto total_pad = checkedAdd(pads[i], pads[input_shape.dim_size() + i]);
    if (ctx.getInputType(0)->tensor_type().shape().dim(i).has_dim_value()) {
      newdim->set_dim_value(checkedAdd(ctx.getInputType(0)->tensor_type().shape().dim(i).dim_value(), total_pad));
    } else if (total_pad == 0) {
      *newdim = input_shape.dim(i);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    2,
    generated::ai_onnx::Pad_ver2().TypeAndShapeInferenceFunction(PadShapeInference_opset2));

static void GatherNDShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // Shape inference
  if (!hasNInputShapes(ctx, 2)) {
    // cannot proceed with shape or rank inference
    return;
  }

  const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto data_rank = data_shape.dim_size();

  const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
  const auto indices_rank = indices_shape.dim_size();

  if (data_rank < 1 || indices_rank < 1) {
    fail_shape_inference(
        "Both `data` and `indices` input tensors in GatherND op "
        "need to have rank larger than 0.");
  }

  // cannot ascertain if the input shapes are valid if shape of
  // `indices` is missing last dimension value so return at this point
  if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
    return;
  }

  const auto last_index_dimension = indices_shape.dim(indices_rank - 1).dim_value();

  if (last_index_dimension < 0) {
    fail_shape_inference("Last dimension of `indices` input tensor in GatherND op must not be negative");
  }
  if (last_index_dimension > data_rank) {
    fail_shape_inference(
        "Last dimension of `indices` input tensor in GatherND op "
        "must not be larger than the rank of `data` tensor");
  }

  for (int i = 0; i < indices_rank - 1; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
  }

  for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
    *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    11,
    generated::ai_onnx::GatherND_ver11().TypeAndShapeInferenceFunction(GatherNDShapeInference_opset11));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    14,
    generated::ai_onnx::Identity_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void WhereShapeInference_opset9(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
  if (hasNInputShapes(ctx, 3)) {
    std::vector<const TensorShapeProto*> shapes;
    shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
    shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
    multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    9,
    generated::ai_onnx::Where_ver9().TypeAndShapeInferenceFunction(WhereShapeInference_opset9));

static void PadShapeInference_opset13(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  // Shape inference needs the input data shape
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
  const auto input_rank = input_shape.dim_size();

  // Infer output shape if 'pads' tensor is available
  const auto pads_initializer = ctx.getInputData(1);
  if (nullptr != pads_initializer) {
    if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
      fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
    }

    const auto pads_data = ParseData<int64_t>(pads_initializer);
    if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
      fail_shape_inference("Pads has incorrect number of values");
    }

    auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    for (int i = 0; i < input_rank; ++i) {
      const auto& input_dim = input_shape.dim(i);
      auto output_dim = output_shape->add_dim();
      const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
      if (input_dim.has_dim_value()) {
        output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
      } else if (total_pad == 0) {
        *output_dim = input_dim;
      }
    }
  } else {
    // Infer output shapes' rank in any case
    auto output_shape_0 = getOutputShape(ctx, 0);
    for (int i = 0; i < input_rank; ++i) {
      output_shape_0->add_dim();
    }
  }
  return;
}

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    13,
    generated::ai_onnx::Pad_ver13().TypeAndShapeInferenceFunction(PadShapeInference_opset13));

ONNX_OPERATOR_SET_SCHEMA(Pad, 18, generated::ai_onnx::Pad_ver18().FillUsing(PadDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    16,
    generated::ai_onnx::Identity_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void ReshapeShapeInference_opset14(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  bool found;
  TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
  if (!found) {
    return;
  }

  int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

  // Iterate through targetShape, adding dimensions in the outputShape
  // TensorProto. If the targetShape dimension is -1, we do not set the
  // dimension value in this iteration, but we record the Dimension. If
  // targetShape dimension is 0, we attempt to propagate the dimension
  // value/param. If the value cannot be inferred, we set the flag in
  // the unresolveZeros vector. If targetShape dimension is positive, we
  // set the dimension value in the outputShape. We track the product of
  // the dimensions we are setting outputShape in the outputProduct
  // variable. The outputProduct will potentially be used for inferring
  // a dimension marked -1.
  auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  TensorShapeProto::Dimension* negativeOneDim = nullptr;
  const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
  std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
  int64_t outputProduct = 1;
  bool outputProductValid = true;
  for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
    // Add a new dimension to outputShape
    auto new_dim = outputShape->add_dim();
    if (targetShapeProto.dim(i).has_dim_param()) {
      // There is a tricky edge case here. It is possible that the value of
      // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
      // symbol can be erroneous. This should be a very rare scenario and in such a
      // case an option is to turn off data propagation during shape inference.
      new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
      outputProductValid = false;
    } else {
      if (!targetShapeProto.dim(i).has_dim_value()) {
        outputProductValid = false;
        // treat this dim as unknown dim
        continue;
      }

      const auto dim_value = targetShapeProto.dim(i).dim_value();

      if (dim_value == -1) {
        // Check if multiple -1's. If not, set negativeOneDim, marking
        // this dimension to potentially be filled in later.
        if (negativeOneDim) {
          fail_shape_inference("Target shape may not have multiple -1 dimensions.");
        }
        negativeOneDim = new_dim;
      } else if (dim_value == 0) {
        // Check if data input has a shape and if the index i is within
        // its bounds. If these conditions are satisfied, any dimension
        // value/param should be propagated. If dimension value cannot be
        // inferred, set the corresponding  unresolvedZeros flag to true.
        // If allowzero is set however, do not propagate values, since output
        // dimension is explicitly zero.
        if (allowzero == 0) {
          unresolvedZeros[i] = true;
          if (dataInputTensorType.has_shape()) {
            if (i >= dataInputTensorType.shape().dim_size()) {
              fail_shape_inference("Invalid position of 0.");
            }
            if (dataInputTensorType.shape().dim(i).has_dim_value()) {
              const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
              new_dim->set_dim_value(input_dim_value);
              checked_mul_into(outputProduct, input_dim_value);
              unresolvedZeros[i] = false;
            } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
              new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
            }
          }
        } else {
          new_dim->set_dim_value(dim_value);
          checked_mul_into(outputProduct, dim_value);
        }
      } else if (dim_value > 0) {
        // Set the dimension value to dim_value
        new_dim->set_dim_value(dim_value);
        checked_mul_into(outputProduct, dim_value);
      } else {
        // Check if value is less than -1; fail if so
        fail_shape_inference("Invalid dimension value: ", dim_value);
      }
    }
  }
  // If negativeOneDim has been set, we attempt to infer its value. This
  // can be done if all dimension values for the data input tensor shape
  // are known other than the ones corresponding to unresolvedZeros
  // flags.
  if (negativeOneDim && outputProductValid) {
    // First, attempt to compute product of data input shape dimensions
    // that are not marked by unresolvedZeros. If not possible, set the
    // inputProductValid flag to false.
    if (!outputProduct) {
      fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
    }
    int64_t inputProduct = 1;
    bool inputProductValid = true;
    if (!dataInputTensorType.has_shape()) {
      inputProductValid = false;
    } else {
      for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
        if (dataInputTensorType.shape().dim(i).has_dim_value()) {
          checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
        } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
          inputProductValid = false;
          break;
        }
      }
    }
    if (inputProductValid) {
      if (inputProduct % outputProduct != 0) {
        fail_shape_inference("Dimension could not be inferred: incompatible shapes");
      }
      negativeOneDim->set_dim_value(inputProduct / outputProduct);
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    14,
    generated::ai_onnx::Reshape_ver14().TypeAndShapeInferenceFunction(ReshapeShapeInference_opset14));

static void ShapeDataPropagation_opset21(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

static void ShapeShapeInference_opset21(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    21,
    generated::ai_onnx::Shape_ver21()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset21)
        .PartialDataPropagationFunction(ShapeDataPropagation_opset21));

static void ShapeDataPropagation_opset19(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

static void ShapeShapeInference_opset19(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    19,
    generated::ai_onnx::Shape_ver19()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset19)
        .PartialDataPropagationFunction(ShapeDataPropagation_opset19));

static void ShapeDataPropagation_opset15(DataPropagationContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
    int64_t rank = static_cast<int64_t>(input_shape.dim_size());
    int64_t start = getAttribute(ctx, "start", 0);
    if (start < 0)
      start += rank;
    start = (start < 0) ? 0 : (start > rank) ? rank : start;
    int64_t end = getAttribute(ctx, "end", rank);
    if (end < 0)
      end += rank;
    end = (end < 0) ? 0 : (end > rank) ? rank : end;
    TensorShapeProto output_shape;
    for (int64_t d = start; d < end; ++d) {
      *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
    }
    ctx.addOutputData(0, std::move(output_shape));
  }
}

static void ShapeShapeInference_opset15(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  auto output_length = output_shape->add_dim();

  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
  int64_t start = getAttribute(ctx, "start", 0);
  if (start < 0)
    start += rank;
  start = (start < 0) ? 0 : (start > rank) ? rank : start;
  int64_t end = getAttribute(ctx, "end", rank);
  if (end < 0)
    end += rank;
  end = (end < 0) ? 0 : (end > rank) ? rank : end;
  output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    15,
    generated::ai_onnx::Shape_ver15()
        .TypeAndShapeInferenceFunction(ShapeShapeInference_opset15)
        .PartialDataPropagationFunction(ShapeDataPropagation_opset15));

static void SizeDataPropagation_opset21(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void SizeShapeInference_opset21(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    21,
    generated::ai_onnx::Size_ver21()
        .TypeAndShapeInferenceFunction(SizeShapeInference_opset21)
        .PartialDataPropagationFunction(SizeDataPropagation_opset21));

static void SizeDataPropagation_opset19(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void SizeShapeInference_opset19(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    19,
    generated::ai_onnx::Size_ver19()
        .TypeAndShapeInferenceFunction(SizeShapeInference_opset19)
        .PartialDataPropagationFunction(SizeDataPropagation_opset19));

static void SizeDataPropagation_opset13(DataPropagationContext& ctx) {
  const auto input_data = ctx.getInputData(0);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
    ctx.addOutputData(0, std::move(tsp));
  }
}

static void SizeShapeInference_opset13(InferenceContext& ctx) {
  ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
}

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    13,
    generated::ai_onnx::Size_ver13()
        .TypeAndShapeInferenceFunction(SizeShapeInference_opset13)
        .PartialDataPropagationFunction(SizeDataPropagation_opset13));

static void CompressShapeInference_opset11(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  auto axisAttr = ctx.getAttribute("axis");
  if (hasInputShape(ctx, 0)) {
    const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
    int r = indices_shape.dim_size();
    if (r < 1) {
      fail_shape_inference("Indices tensor must have rank >= 1");
    }
    if (axisAttr) {
      int axis = static_cast<int>(axisAttr->i());
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
      }
      if (axis < 0) {
        axis += r;
      }
      TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      for (int i = 0; i < indices_shape.dim_size(); i++) {
        auto dim = shape->add_dim();
        if (i != axis) {
          *dim = indices_shape.dim(i);
        }
      }
    }
  }
  if (!axisAttr) {
    updateOutputShape(ctx, 0, {Dim()});
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    11,
    generated::ai_onnx::Compress_ver11().TypeAndShapeInferenceFunction(CompressShapeInference_opset11));

static void OneHotShapeInference_opset11(InferenceContext& ctx) {
  oneHotShapeInference(ctx, 11);
}

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    11,
    generated::ai_onnx::OneHot_ver11().TypeAndShapeInferenceFunction(OneHotShapeInference_opset11));

static void ReverseSequenceShapeInference_opset10(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto& first_input_shape = getInputShape(ctx, 0);
  if (first_input_shape.dim_size() < 2) {
    fail_shape_inference("'input' must have rank >= 2");
  }
  auto& seq_len_input_shape = getInputShape(ctx, 1);
  if (seq_len_input_shape.dim_size() != 1) {
    fail_shape_inference("'sequence_lens' must have rank of 1");
  }

  propagateShapeFromInputToOutput(ctx, 0, 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    10,
    generated::ai_onnx::ReverseSequence_ver10().TypeAndShapeInferenceFunction(ReverseSequenceShapeInference_opset10));

static void UniqueShapeInference_opset11(InferenceContext& ctx) {
  // Type inference
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const TypeProto* xTensorProto = ctx.getInputType(0);
  TypeProto* yTensorProto = ctx.getOutputType(0);
  TypeProto* indicesTensorProto = nullptr;
  TypeProto* inverseIndicesTensorProto = nullptr;
  TypeProto* countsTensorProto = nullptr;

  auto num_outputs = ctx.getNumOutputs();
  if (num_outputs >= 2) {
    indicesTensorProto = ctx.getOutputType(1);
    updateOutputElemType(ctx, 1, TensorProto::INT64);
    indicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 3) {
    inverseIndicesTensorProto = ctx.getOutputType(2);
    updateOutputElemType(ctx, 2, TensorProto::INT64);
    inverseIndicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  if (num_outputs >= 4) {
    countsTensorProto = ctx.getOutputType(3);
    updateOutputElemType(ctx, 3, TensorProto::INT64);
    countsTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  }

  auto axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
  } else {
    int axis = static_cast<int>(axisAttr->i());
    if (!xTensorProto->tensor_type().has_shape()) {
      return;
    }
    const TensorShapeProto& input_shape = xTensorProto->tensor_type().shape();
    int rank = input_shape.dim_size();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank) {
      fail_shape_inference("Invalid value for attribute axis");
    }
    for (int i = 0; i < input_shape.dim_size(); i++) {
      auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
      if (i != axis) {
        *dim = input_shape.dim(i);
      }
    }
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    generated::ai_onnx::Unique_ver11().TypeAndShapeInferenceFunction(UniqueShapeInference_opset11));

} // namespace ONNX_NAMESPACE
