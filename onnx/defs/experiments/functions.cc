// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/common/constants.h"
#include "onnx/common/model_helpers.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;
using SupportType = ONNX_NAMESPACE::OpSchema::SupportType;

static const char* mvn_ver9_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    9,
    OpSchema()
        .SetSupportLevel(SupportType::COMMON)
        .SetDoc(mvn_ver9_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Attr(
            "axes",
            "A list of integers, along which to reduce. The default is to reduce over "
            "all the dimensions of the input tensor. Use [0,2,3] (without C axis for "
            "N-D cases) for calculating means and variances along channels. Two "
            "variables with the same C-coordinate are associated "
            "with the same mean and variance.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to all numeric tensors.")
        .FunctionBody(FunctionBodyHelper::Define(
            {// nodes: {outputs, op, inputs, attributes}
             FunctionBodyHelper::Const<float>("Exponent", 2.0f),
             FunctionBodyHelper::Const<float>("Epsilon", float(1e-9)),
             {{"X_RM"}, "ReduceMean", {"X"}, {MakeRefAttribute("axes", AttributeProto::INTS)}},
             {{"EX_squared"}, "Pow", {"X_RM", "Exponent"}},
             {{"X_squared"}, "Pow", {"X", "Exponent"}},
             {{"E_Xsquared"}, "ReduceMean", {"X_squared"}, {MakeRefAttribute("axes", AttributeProto::INTS)}},
             {{"Variance"}, "Sub", {"E_Xsquared", "EX_squared"}},
             {{"STD"}, "Sqrt", {"Variance"}},
             {{"X_variance"}, "Sub", {"X", "X_RM"}},
             {{"Processed_STD"}, "Add", {"STD", "Epsilon"}},
             {{"Y"}, "Div", {"X_variance", "Processed_STD"}}})));

} // namespace ONNX_NAMESPACE
