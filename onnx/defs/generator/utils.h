// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

template <typename T>
int64_t compute_output_dim_for_range(const TensorProto* start, const TensorProto* limit, const TensorProto* delta) {
  if (!start->dims().empty() || !limit->dims().empty() || !delta->dims().empty()) {
    fail_shape_inference("Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto start_data = ParseData<T>(start);
  const auto limit_data = ParseData<T>(limit);
  const auto delta_data = ParseData<T>(delta);

  if constexpr (std::is_integral_v<T>) {
    const auto start_value = start_data[0];
    const auto limit_value = limit_data[0];
    const auto delta_value = delta_data[0];
    if (delta_value == 0) {
      fail_shape_inference("Input 'delta' of 'Range' must not be zero");
    }
    const bool increasing = delta_value > 0;
    if ((increasing && limit_value <= start_value) || (!increasing && limit_value >= start_value)) {
      return 0;
    }

    // Unsigned subtraction gives the exact distance across the signed boundary,
    // and can also represent the magnitude of the minimum negative step.
    using U = std::make_unsigned_t<T>;
    const U distance = increasing ? static_cast<U>(limit_value) - static_cast<U>(start_value)
                                  : static_cast<U>(start_value) - static_cast<U>(limit_value);
    const U step = increasing ? static_cast<U>(delta_value) : U{0} - static_cast<U>(delta_value);
    const U n = distance / step + (distance % step != 0);
    if (n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      fail_shape_inference("The output dimension of 'Range' exceeds int64");
    }
    return static_cast<int64_t>(n);
  }

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  n = std::max<int64_t>(n, 0);

  return n;
}

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE::defs::generator::utils {

void ConstantOpInference(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset25(InferenceContext& ctx);
void EyeLikeShapeInference_opset22(InferenceContext& ctx);
void RandomUniformShapeInference_opset22(InferenceContext& ctx);
void RandomNormalShapeInference_opset22(InferenceContext& ctx);
void RandomUniformLikeShapeInference_opset22(InferenceContext& ctx);
void RandomNormalLikeShapeInference_opset22(InferenceContext& ctx);
void MultinomialShapeInference_opset22(InferenceContext& ctx);
bool BuildFunctionBodyRange27(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void RangeShapeInference_opset27(InferenceContext& ctx);
bool BernoulliFunctionBody_opset22(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void BernoulliShapeInference_opset22(ONNX_NAMESPACE::InferenceContext& ctx);
bool BernoulliFunctionBody_opset15(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto);
void BernoulliShapeInference_opset15(ONNX_NAMESPACE::InferenceContext& ctx);
void MultinomialShapeInference_opset7(InferenceContext& ctx);
void RandomNormalLikeShapeInference_opset1(InferenceContext& ctx);
void RandomUniformLikeShapeInference_opset1(InferenceContext& ctx);
void RandomNormalShapeInference_opset1(InferenceContext& ctx);
void RandomUniformShapeInference_opset1(InferenceContext& ctx);
void EyeLikeShapeInference_opset9(InferenceContext& ctx);
void ConstantShapeInference_opset1(InferenceContext& ctx);
void ConstantShapeInference_opset9(InferenceContext& ctx);
void ConstantShapeInference_opset11(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset24(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset23(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset21(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset20(InferenceContext& ctx);
void ConstantOfShapeShapeInference_opset9(InferenceContext& ctx);
void RangeShapeInference_opset11(InferenceContext& ctx);

} // namespace ONNX_NAMESPACE::defs::generator::utils
