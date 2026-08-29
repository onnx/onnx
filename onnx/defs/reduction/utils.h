// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

// Constants used to indicate value returned by reduction of an empty set of values.
constexpr const char* EMPTY_ZERO = "0";
constexpr const char* EMPTY_ONE = "1";
constexpr const char* EMPTY_UNDEFINED = "undefined";
constexpr const char* EMPTY_MIN =
    "minus infinity (if supported by the datatype) or the minimum value of the data type otherwise";
constexpr const char* EMPTY_MAX =
    "plus infinity (if supported by the datatype) or the maximum value of the data type otherwise";
constexpr const char* EMPTY_MINUS_INF = "minus infinity (if supported by the datatype) or undefined otherwise";

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
    data_exp = Exp (data_double)
    reduced_sum = ReduceSum<keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes>(data_exp, axes)
    reduced_double = Log (reduced_sum)
    reduced = CastLike(reduced_double, data)
  }
  )ONNX";

std::function<void(OpSchema&)> ReduceOpGenerator(
    const char* name,
    const char* empty_value,
    bool supports_8bit_datatypes = false,
    bool axes_input = false,
    const char* func_body = nullptr,
    const ContextDependentFunctionBodyBuilder& function_builder = nullptr,
    bool supports_boolean_datatype = false,
    bool float_types_only = false);

inline std::function<void(OpSchema&)> ReduceOpDynamicAxes(const char* name, const char* empty_value) {
  return ReduceOpGenerator(name, empty_value, false, true, nullptr, nullptr, false);
}

inline std::function<void(OpSchema&)>
ReduceFunctionOp(const char* name, const char* empty_value, const char* func_body) {
  return ReduceOpGenerator(name, empty_value, false, true, func_body);
}

// Same as ReduceFunctionOp, but restricts T to float types. Log and Exp are only defined for
// float types, so ops whose function body routes through them cannot support integers.
inline std::function<void(OpSchema&)>
ReduceFunctionOpFloatOnly(const char* name, const char* empty_value, const char* func_body) {
  return ReduceOpGenerator(name, empty_value, false, true, func_body, nullptr, false, true);
}

} // namespace ONNX_NAMESPACE
