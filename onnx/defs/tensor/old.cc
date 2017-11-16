// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;
using namespace onnx;

OPERATOR_SCHEMA(Split)
    .SinceVersion(1)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split", "T")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')", "T")
    .Output(0, "outputs...", "One or more outputs forming list of tensors after splitting", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.")
    .Attr("axis",
          "Which axis to split on",
          AttrType::INT)
    .Attr("split",
          "length of each output",
          AttrType::INTS)
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC");
