// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;

namespace onnx {

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Computes the `{name} than` elementwise logical operation between `left` and `right` input tensor. 
The result is a tensor of type integer in which `0` mean false and `1` mean true.)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.NumInputs(2);
        schema.NumOutputs(1);
        schema.SetDoc(doc);
        schema.Input(0, "left", "Left input tensor for the logical operator.", "T");
        schema.Input(1, "right", "Right input tensor for the logical operator.", "T");
        schema.Output(0, "output", "Result tensor of type `int`, 0 mean False and 1 mean True.", "T");
        schema.TypeConstraint("T", { "tensor(int32)" }, "Constrain input and output types to int tensor.");
    };
}

}  // namespace onnx
