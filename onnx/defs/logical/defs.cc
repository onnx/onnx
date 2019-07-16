// Copyright (c) ONNX Project Contributors.
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
            "Constrains input types to all numeric tensors.")
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
            "Constrains input types to all numeric tensors.")
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

static const char* BitShift_ver11_doc = R"DOC(
Bitwise shift operator performs element-wise operation. For each input element, if the
 attribute "direction" is "RIGHT", this operator moves its binary representation toward
 the right side so that the input value is effectively decreased. If the attribute "direction"
 is "LEFT", bits of binary representation moves toward the left side, which results the
 increase of its actual value. The input X is the tensor to be shifted and another input
 Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
 and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
 X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].
 
 Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
 not necessarily identical.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BitShift,
    11,
    OpSchema()
        .SetDoc(std::string(BitShift_ver11_doc) + GenerateBroadcastingDocMul())
        .Input(0, "X", "First operand, input to be shifted.", "T")
        .Input(1, "Y", "Second operand, amounts of shift.", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"},
            "Constrain input and output types to integer tensors.")
        .Attr(
            "direction",
            "Direction of moving bits. It can be either \"RIGHT\" (for right shift) "
            "or \"LEFT\" (for left shift).",
            AttributeProto::STRING)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

}  // namespace ONNX_NAMESPACE
