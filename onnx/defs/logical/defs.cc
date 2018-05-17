// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

inline void logicalOpInference(InferenceContext& ctx) {
	updateOutputElemType(ctx, 0, TensorProto::BOOL);
	if (hasInputShape(ctx, 0)) {
		propagateShapeFromInputToOutput(ctx, 0, 0);
	}
}

std::function<void(OpSchema&)> BinaryLogicDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Returns the tensor resulted from performing the `{name}` logical operation
elementwise on the input tensors `A` and `B`.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details
please check [the doc](Broadcasting.md).
)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Input(0, "A", "First input operand for the logical operator.", "T");
        schema.Input(1, "B", "Second input operand for the logical operator.", "T");
		schema.Output(0, "C", "Result tensor.", "T1");
        schema.SinceVersion(7);
		schema.TypeAndShapeInferenceFunction(logicalOpInference);
    };
}

ONNX_OPERATOR_SCHEMA(And)
    .FillUsing(BinaryLogicDocGenerator("and"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Or)
    .FillUsing(BinaryLogicDocGenerator("or"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Xor)
    .FillUsing(BinaryLogicDocGenerator("xor"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Greater)
    .FillUsing(BinaryLogicDocGenerator("greater"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Less)
    .FillUsing(BinaryLogicDocGenerator("less"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Equal)
    .FillUsing(BinaryLogicDocGenerator("equal"))
    .TypeConstraint("T", { "tensor(bool)", "tensor(int32)", "tensor(int64)" },
                    "Constrains input to integral tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

ONNX_OPERATOR_SCHEMA(Not)
    .SetDoc(R"DOC(
Returns the negation of the input tensor element-wise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input/output to boolean tensors.")
	.TypeAndShapeInferenceFunction(logicalOpInference);

}  // namespace ONNX_NAMESPACE
