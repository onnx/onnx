// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
#include <functional>

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> ReduceDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Attr("axes",
                    "A list of integers, along which to reduce.",
                    AttributeProto::INTS,
                    OPTIONAL);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttributeProto::INT,
                    static_cast<int64_t>(1));
        schema.Input(0, "data", "An input tensor.", "T");
        schema.Output(0, "reduced", "Reduced output tensor.", "T");
        schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");
    };
}
  
OPERATOR_SCHEMA(ReduceMax)
    .FillUsing(ReduceDocGenerator("max"));

OPERATOR_SCHEMA(ReduceMin)
    .FillUsing(ReduceDocGenerator("min"));

OPERATOR_SCHEMA(ReduceSum)
    .FillUsing(ReduceDocGenerator("sum"));

OPERATOR_SCHEMA(ReduceSumSquare)
    .FillUsing(ReduceDocGenerator("sum square"));

OPERATOR_SCHEMA(ReduceMean)
    .FillUsing(ReduceDocGenerator("mean"));

OPERATOR_SCHEMA(ReduceProd)
    .FillUsing(ReduceDocGenerator("product"));

OPERATOR_SCHEMA(ReduceLogSum)
    .FillUsing(ReduceDocGenerator("log sum"));

OPERATOR_SCHEMA(ReduceLogSumExp)
    .FillUsing(ReduceDocGenerator("log sum exponent"));

OPERATOR_SCHEMA(ReduceL1)
    .FillUsing(ReduceDocGenerator("L1 norm"));

OPERATOR_SCHEMA(ReduceL2)
    .FillUsing(ReduceDocGenerator("L2 norm"));


}  // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> ArgReduceDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Attr("axis",
                    "The axis in which to compute the arg indices",
                    AttributeProto::INT,
                    OPTIONAL);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttributeProto::INT,
                    static_cast<int64_t>(1));
        schema.Input(0, "data", "An input tensor.", "T");
        schema.Output(0, "reduced", "Reduced output tensor with integer data type.", "tensor(int32)");
        schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.");
    };
}

OPERATOR_SCHEMA(ArgMax)
    .FillUsing(ArgReduceDocGenerator("max"));

OPERATOR_SCHEMA(ArgMin)
    .FillUsing(ArgReduceDocGenerator("min"));

}  // namespace ONNX_NAMESPACE
