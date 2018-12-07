// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

inline void unaryLogicalOpInference(InferenceContext& ctx) {
    updateOutputElemType(ctx, 0, TensorProto::BOOL);
    if (hasInputShape(ctx, 0)) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
    }
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str());
        schema.SetDoc(doc);
        schema.Input(0, "A", "First input operand for the logical operator.", "T");
        schema.Input(1, "B", "Second input operand for the logical operator.", "T");
        schema.Output(0, "C", "Result tensor.", "T1");
        schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        });
    };
}

ONNX_OPERATOR_SET_SCHEMA(
    And,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("and"))
        .TypeConstraint(
            "T",
            {"tensor(bool)"},
            "Constrains input to boolean tensor.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Or,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("or"))
        .TypeConstraint(
            "T",
            {"tensor(bool)"},
            "Constrains input to boolean tensor.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Xor,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("xor"))
        .TypeConstraint(
            "T",
            {"tensor(bool)"},
            "Constrains input to boolean tensor.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Greater,
    9,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("greater"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrains input to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Less,
    9,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("less"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrains input to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

ONNX_OPERATOR_SET_SCHEMA(
    Equal,
    7,
    OpSchema()
        .FillUsing(BinaryLogicDocGenerator("equal"))
        .TypeConstraint(
            "T",
            {"tensor(bool)", "tensor(int32)", "tensor(int64)"},
            "Constrains input to integral tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrains output to boolean tensor."));

static const char* Not_ver1_doc = R"DOC(
Returns the negation of the input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Not,
    1,
    OpSchema()
        .SetDoc(Not_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(bool)"},
            "Constrains input/output to boolean tensors.")
        .TypeAndShapeInferenceFunction(unaryLogicalOpInference));

}  // namespace ONNX_NAMESPACE
