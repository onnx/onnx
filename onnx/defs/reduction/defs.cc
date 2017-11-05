// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
#include <functional>

using AttrType = onnx::OpSchema::AttrType;

namespace onnx {

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
                    AttrType::INTS);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttrType::INT);
        schema.Input(0, "data", "An input tensor.", "T");
        schema.Output(0, "reduced", "Reduced output tensor.", "T");
        schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, 
            "Constrain input and output types to float tensors.");
    };
}
  
OPERATOR_SCHEMA(ReduceMax)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("max"));

OPERATOR_SCHEMA(ReduceMin)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("min"));

OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("sum"));

OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("mean"));

OPERATOR_SCHEMA(ReduceProd)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("product"));

OPERATOR_SCHEMA(ReduceLogSumExp)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("log sum exponent"));

}  // namespace onnx

namespace onnx {

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
                    AttrType::INT);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttrType::INT);
        schema.Input(0, "data", "An input tensor.", "T");
        schema.Output(0, "reduced", "Reduced output tensor with integer data type.", "T");
        schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.");
    };
}

OPERATOR_SCHEMA(ArgMax)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ArgReduceDocGenerator("max"));

OPERATOR_SCHEMA(ArgMin)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ArgReduceDocGenerator("min"));

}  // namespace onnx
