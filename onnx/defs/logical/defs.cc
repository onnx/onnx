// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;

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
        schema.Attr("broadcast", "Enable broadcasting", AttrType::INT);
        schema.Attr("axis", "If set, defines the broadcast dimensions.",
                    AttrType::INT);
        schema.Input(0, "A", "Left input tensor for the logical operator.", "T");
        schema.Input(1, "B", "Right input tensor for the logical operator.", "T");
        schema.Output(0, "C", "Result tensor.", "T1");
    };
}

OPERATOR_SCHEMA(And)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("and"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Or)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("or"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Xor)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("xor"))
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input to boolean tensor.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Greater)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("greater"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Less)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("less"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Equal)
    .NumInputs(2)
    .NumOutputs(1)
    .FillUsing(BinaryLogicDocGenerator("equal"))
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrains input to float tensors.")
    .TypeConstraint("T1", { "tensor(bool)" },
                    "Constrains output to boolean tensor.");

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Returns the negation of the input tensor element-wise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(bool)" },
                    "Constrains input/output to boolean tensors.");

}  // namespace onnx
