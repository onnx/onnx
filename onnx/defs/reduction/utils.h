// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

// Function bodies shared between the current schemas in defs.cc and the older ones in old.cc.
static constexpr const char* reduce_log_sum_func_body = R"ONNX(
  {
    reduced_sum = ReduceSum<keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes>(data, axes)
    reduced = Log (reduced_sum)
  }
  )ONNX";

static constexpr const char* reduce_log_sum_exp_func_body = R"ONNX(
  {
    data_double = Cast<to = 11>(data)
    data_exp = Exp(data_double)
    reduced_sum = ReduceSum<keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes>(data_exp, axes)
    reduced_double = Log(reduced_sum)
    reduced = CastLike(reduced_double, data)
  }
  )ONNX";

std::function<void(OpSchema&)> ReduceOpGenerator(
    const char* func_body = nullptr,
    const ContextDependentFunctionBodyBuilder& function_builder = nullptr);

inline std::function<void(OpSchema&)> ReduceOpDynamicAxes() {
  return ReduceOpGenerator(nullptr, nullptr);
}

inline std::function<void(OpSchema&)> ReduceFunctionOp(const char* func_body) {
  return ReduceOpGenerator(func_body);
}

// Same as ReduceFunctionOp, but restricts T to float types. Log and Exp are only defined for
// float types, so ops whose function body routes through them cannot support integers.
inline std::function<void(OpSchema&)> ReduceFunctionOpFloatOnly(const char* func_body) {
  return ReduceOpGenerator(func_body, nullptr);
}

} // namespace ONNX_NAMESPACE
