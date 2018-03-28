// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

const char* kBroadcastDoc_old = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of size 1 (a scalar value), or having its shape as a
contiguous subset of the first tensor's shape. The starting of the mutually
equal shape is specified by the argument "axis", and if it is not set, suffix
matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
)DOC";

std::function<void(OpSchema&)> MathDocGenerator_old(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc_old);
    schema.SetDoc(doc);
    schema.Attr(
        "broadcast",
        "Pass 1 to enable broadcasting",
        AttributeProto::INT,
        static_cast<int64_t>(0));

    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    schema.Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.",
        AttributeProto::INT,
        OPTIONAL);
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.",
        "T");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.",
        "T");
    schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(Add).FillUsing(MathDocGenerator_old("addition"));

ONNX_OPERATOR_SCHEMA(Sub).FillUsing(MathDocGenerator_old("subtraction"));

ONNX_OPERATOR_SCHEMA(Mul).FillUsing(MathDocGenerator_old("multiplication"));

ONNX_OPERATOR_SCHEMA(Div).FillUsing(MathDocGenerator_old("division"));
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SCHEMA(Neg)
    .SetDoc(R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Abs)
    .SetDoc(R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Reciprocal)
    .SetDoc(R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Floor)
    .SetDoc(R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Ceil)
    .SetDoc(R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sqrt)
    .SetDoc(R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Relu)
    .SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(LeakyRelu)
    .Attr(
        "alpha",
        "Coefficient of leakage default to 0.01.",
        AttributeProto::FLOAT,
        0.01f)
    .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Selu)
    .Attr(
        "alpha",
        "Coefficient of SELU default to 1.6732.",
        AttributeProto::FLOAT,
        1.6732f)
    .Attr(
        "gamma",
        "Coefficient of SELU default to 1.0507.",
        AttributeProto::FLOAT,
        1.0507f)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .SetDoc(R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Elu)
    .Attr(
        "alpha",
        "Coefficient of ELU default to 1.0.",
        AttributeProto::FLOAT,
        1.0f)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .SetDoc(R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC")
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D input tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Exp)
    .SetDoc(R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor", "T")
    .Output(
        0,
        "output",
        "The exponential of the input tensor computed "
        "element-wise",
        "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Log)
    .SetDoc(R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC")
    .Input(0, "input", "Input tensor", "T")
    .Output(
        0,
        "output",
        "The natural log of the input tensor computed "
        "element-wise",
        "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Tanh)
    .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC")
    .Input(0, "input", "1-D input tensor", "T")
    .Output(
        0,
        "output",
        "The hyperbolic tangent values of the input tensor "
        "computed element-wise",
        "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(PRelu)
    .SetDoc(R"DOC(

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Input(
        1,
        "slope",
        "Slope tensor. If `Slope` is of size 1, the value is shared"
        "across different channels",
        "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sigmoid)
    .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(HardSigmoid)
    .Attr("alpha", "Value of alpha default to 0.2", AttributeProto::FLOAT, 0.2f)
    .Attr("beta", "Value of beta default to 0.5", AttributeProto::FLOAT, 0.5f)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .SetDoc(R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Max)
    .SetDoc(R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
    .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Min)
    .SetDoc(R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
    .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sum)
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
    .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Mean)
    .SetDoc(R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Mean.", "T", OpSchema::Variadic)
    .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Clip)
    .SetDoc(R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC")
    .Attr(
        "min",
        "Minimum value, under which element is replaced by min",
        AttributeProto::FLOAT,
        OPTIONAL)
    .Attr(
        "max",
        "Maximum value, above which element is replaced by max",
        AttributeProto::FLOAT,
        OPTIONAL)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .Input(0, "input", "Input tensor whose elements to be clipped", "T")
    .Output(0, "output", "Output tensor with clipped input elements", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Gemm)
    .SetDoc(R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has dimension (M X K)
, input tensor B has dimension (K X N), input tensor C and output tensor Y have
dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. If A can be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC")
    .Input(0, "A", "Input tensor A", "T")
    .Input(1, "B", "Input tensor B", "T")
    .Input(2, "C", "Input tensor C, can be inplace.", "T")
    .Output(0, "Y", "Output tensor.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "transA",
        "Whether A should be transposed",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Attr(
        "transB",
        "Whether B should be transposed",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Attr(
        "broadcast",
        "Whether C should be broadcasted",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Attr(
        "alpha",
        "Scalar multiplier for the product of input tensors A * B",
        AttributeProto::FLOAT,
        1.0f)
    .Attr(
        "beta",
        "Scalar multiplier for input tensor C",
        AttributeProto::FLOAT,
        1.0f);