// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

namespace onnx {

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.Attr(
        "broadcast",
        "Enable broadcasting",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "axis",
        "If set, defines the broadcast dimensions.",
        AttributeProto::INT,
        OPTIONAL);
    schema.Input(0, "A", "Left input tensor for the logical operator.", "T");
    schema.Input(1, "B", "Right input tensor for the logical operator.", "T");
    schema.Output(0, "C", "Result tensor.", "T1");
  };
}

OPERATOR_SCHEMA(And)
    .FillUsing(BinaryLogicDocGenerator("and"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_BOOL},
        "Constrains input to boolean tensor.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Or)
    .FillUsing(BinaryLogicDocGenerator("or"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_BOOL},
        "Constrains input to boolean tensor.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Xor)
    .FillUsing(BinaryLogicDocGenerator("xor"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_BOOL},
        "Constrains input to boolean tensor.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Greater)
    .FillUsing(BinaryLogicDocGenerator("greater"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_FLOAT16,
         DataType::Tensor_FLOAT,
         DataType::Tensor_DOUBLE},
        "Constrains input to float tensors.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Less)
    .FillUsing(BinaryLogicDocGenerator("less"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_FLOAT16,
         DataType::Tensor_FLOAT,
         DataType::Tensor_DOUBLE},
        "Constrains input to float tensors.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Equal)
    .FillUsing(BinaryLogicDocGenerator("equal"))
    .TypeConstraint(
        "T",
        {DataType::Tensor_FLOAT16,
         DataType::Tensor_FLOAT,
         DataType::Tensor_DOUBLE},
        "Constrains input to float tensors.")
    .TypeConstraint(
        "T1",
        {DataType::Tensor_BOOL},
        "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Not)
    .SetDoc(R"DOC(
Returns the negation of the input tensor element-wise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {DataType::Tensor_BOOL},
        "Constrains input/output to boolean tensors.");

} // namespace onnx
