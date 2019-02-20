// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/common/constants.h"
#include "onnx/common/model_helpers.h"
#include "onnx/defs/function.h"

namespace ONNX_NAMESPACE {
static Common::Status BuildMVN(std::unique_ptr<FunctionProto>* func_proto) {
  if (nullptr == func_proto) {
    return Common::Status(
        Common::CHECKER,
        Common::INVALID_ARGUMENT,
        "func_proto should not be nullptr.");
  }

  func_proto->reset(new FunctionProto);
  auto& func = **func_proto;

  func = FunctionProtoHelper::Define(
      "MeanVarianceNormalization", // name
      9,
      {"X"}, // inputs
      {"X_MVN"}, // outputs
      {"axes"}, // attributes
      {
          // nodes: {outputs, op, inputs, attributes}
          FunctionProtoHelper::Const<float>("Exponent", 2.0f),
          FunctionProtoHelper::Const<float>("Epsilon", float(1e-9)),
          {{"X_RM"}, "ReduceMean", {"X"}, {{"axes", "$axes:ints"}}},
          {{"EX_squared"}, "Pow", {"X_RM", "Exponent"}},
          {{"X_squared"}, "Pow", {"X", "Exponent"}},
          {{"E_Xsquared"},
           "ReduceMean",
           {"X_squared"},
           {{"axes", "$axes:ints"}}},
          {{"Variance"}, "Sub", {"E_Xsquared", "EX_squared"}},
          {{"STD"}, "Sqrt", {"Variance"}},
          {{"X_variance"}, "Sub", {"X", "X_RM"}},
          {{"Processed_STD"}, "Add", {"STD", "Epsilon"}},
          {{"X_MVN"}, "Div", {"X_variance", "Processed_STD"}},
      });

  return Common::Status::OK();
}

ONNX_FUNCTION_BUILD(
    MeanVarianceNormalization,
    9,
    FunctionBuilder().SetDomain(ONNX_DOMAIN).SetBuildFunction(BuildMVN));

FunctionProto SoftmaxGradFunc = FunctionProtoHelper::Define(
    "SoftmaxGrad", //
    9,
    {"x", "grad_softmax"},
    {"grad_x"},
    {},
    {{{"softmax"}, "Softmax", {"x"}},
     {{"n0"}, "Mul", {"grad_softmax", "softmax"}},
     FunctionProtoHelper::Const("indices", 1.0f),
     {{"n1"}, "Sum", {"n0", "indices"}},
     FunctionProtoHelper::Const<float>("newshape", {-1, 1}),
     {{"n2"}, "Reshape", {"n1", "newshape"}},
     {{"n3"}, "Sub", {"grad_softmax", "n2"}},
     {{"grad_x"}, "Mul", {"n3", "softmax"}}});

// ONNX_FUNCTION_BUILD(
//    SoftmaxGrad,
//    9,
//    FunctionBuilder().SetDomain(ONNX_DOMAIN).SetBuildFunction(BuildSigmoidGrad));

// static Common::Status BuildSigmoidGrad(
//    std::unique_ptr<FunctionProto> * func_proto, ) {}

} // namespace ONNX_NAMESPACE
