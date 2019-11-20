// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator_opset1(
    const char* name,
    const char* description) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The operator computes the {name} ({description}) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the {name} values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{description}", description);
    schema.SetDoc(doc);
    schema.Attr(
        "axis",
        "Describes the axis of the inputs when coerced "
        "to 2D; defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(
        0,
        "input",
        "The input tensor that's coerced into a 2D matrix of size (NxD) "
        "as described above.",
        "T");
    schema.Output(
        0,
        "output",
        "The output values with the same "
        "shape as input tensor (the original size without coercion).",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    1,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator_opset1("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    1,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator_opset1("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    1,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator_opset1(
        "hardmax",
        "1 for the first maximum value, and 0 for all others")));

const char* kBroadcastDoc_old = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
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

std::function<void(OpSchema&)> MathDocGenerator_old_opset6(const char* name) {
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
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    1,
    OpSchema().FillUsing(MathDocGenerator_old("addition")));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    1,
    OpSchema().FillUsing(MathDocGenerator_old("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    1,
    OpSchema().FillUsing(MathDocGenerator_old("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(
    Div,
    1,
    OpSchema().FillUsing(MathDocGenerator_old("division")));

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    6,
    OpSchema().FillUsing(MathDocGenerator_old_opset6("addition")));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    6,
    OpSchema().FillUsing(MathDocGenerator_old_opset6("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    6,
    OpSchema().FillUsing(MathDocGenerator_old_opset6("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(
    Div,
    6,
    OpSchema().FillUsing(MathDocGenerator_old_opset6("division")));

static const char* Pow_ver1_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    1,
    OpSchema()
        .SetDoc(Pow_ver1_doc + std::string(kBroadcastDoc_old))
        .Input(0, "X", "Input tensor of any shape, base of the exponent.", "T")
        .Input(
            1,
            "Y",
            "Input tensor of any shape broadcastable to X shape, "
            "the exponent component.",
            "T")
        .Attr(
            "broadcast",
            "Pass 1 to enable broadcasting",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "axis",
            "If set, defines the broadcast dimensions. See doc for details.",
            AttributeProto::INT,
            OPTIONAL)
        .Output(0, "Z", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Neg_ver1_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    1,
    OpSchema()
        .SetDoc(Neg_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Abs_ver1_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    1,
    OpSchema()
        .SetDoc(Abs_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Reciprocal_ver1_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    1,
    OpSchema()
        .SetDoc(Reciprocal_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Floor_ver1_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    1,
    OpSchema()
        .SetDoc(Floor_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Ceil_ver1_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    1,
    OpSchema()
        .SetDoc(Ceil_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Sqrt_ver1_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    1,
    OpSchema()
        .SetDoc(Sqrt_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Relu_ver1_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    1,
    OpSchema()
        .SetDoc(Relu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* LeakyRelu_ver1_doc = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    1,
    OpSchema()
        .Attr(
            "alpha",
            "Coefficient of leakage default to 0.01.",
            AttributeProto::FLOAT,
            0.01f)
        .SetDoc(LeakyRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Selu_ver1_doc = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    1,
    OpSchema()
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
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .SetDoc(Selu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Elu_ver1_doc = R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    1,
    OpSchema()
        .Attr(
            "alpha",
            "Coefficient of ELU default to 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .SetDoc(Elu_ver1_doc)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Exp_ver1_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    1,
    OpSchema()
        .SetDoc(Exp_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The exponential of the input tensor computed "
            "element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Log_ver1_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    1,
    OpSchema()
        .SetDoc(Log_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The natural log of the input tensor computed "
            "element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Tanh_ver1_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    1,
    OpSchema()
        .SetDoc(Tanh_ver1_doc)
        .Input(0, "input", "1-D input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* PRelu_ver1_doc = R"DOC(

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    1,
    OpSchema()
        .SetDoc(PRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. If `Slope` is of size 1, the value is shared"
            "across different channels",
            "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    6,
    OpSchema()
        .SetDoc(PRelu_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. If `Slope` is of size 1, the value is shared"
            "across different channels",
            "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* PRelu_ver7_doc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    7,
    OpSchema()
        .SetDoc(
            PRelu_ver7_doc +
            GenerateBroadcastingDocUni("tensor slope", "input tensor X"))
        .Input(0, "X", "Input tensor", "T")
        .Input(
            1,
            "slope",
            "Slope tensor. The shape of slope can be smaller then first input X; "
            "if so, its shape must be unidirectional broadcastable to X",
            "T")
        .Output(0, "Y", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sigmoid_ver1_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    1,
    OpSchema()
        .SetDoc(Sigmoid_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* HardSigmoid_ver1_doc = R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    1,
    OpSchema()
        .Attr(
            "alpha",
            "Value of alpha default to 0.2",
            AttributeProto::FLOAT,
            0.2f)
        .Attr(
            "beta",
            "Value of beta default to 0.5",
            AttributeProto::FLOAT,
            0.5f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .SetDoc(HardSigmoid_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Max_ver1_doc = R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    1,
    OpSchema()
        .SetDoc(Max_ver1_doc)
        .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
        .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Min_ver1_doc = R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    1,
    OpSchema()
        .SetDoc(Min_ver1_doc)
        .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
        .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Sum_ver1_doc = R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    1,
    OpSchema()
        .SetDoc(Sum_ver1_doc)
        .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
        .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Mean_ver1_doc = R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    1,
    OpSchema()
        .SetDoc(Mean_ver1_doc)
        .Input(
            0,
            "data_0",
            "List of tensors for Mean.",
            "T",
            OpSchema::Variadic)
        .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr(
            "consumed_inputs",
            "legacy optimization attribute.",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Clip_ver1_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    1,
    OpSchema()
        .SetDoc(Clip_ver1_doc)
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
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
            "Constrain input and output types to float tensors."));

static const char* Gemm_ver1_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    1,
    OpSchema()
        .SetDoc(Gemm_ver1_doc)
        .Input(0, "A", "Input tensor A", "T")
        .Input(1, "B", "Input tensor B", "T")
        .Input(2, "C", "Input tensor C, can be inplace.", "T")
        .Output(0, "Y", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
            "Scalar multiplier for the product of input tensors A * B, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "beta",
            "Scalar multiplier for input tensor C, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f));

static const char* Gemm_ver6_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    6,
    OpSchema()
        .SetDoc(Gemm_ver6_doc)
        .Input(0, "A", "Input tensor A", "T")
        .Input(1, "B", "Input tensor B", "T")
        .Input(2, "C", "Input tensor C", "T")
        .Output(0, "Y", "Output tensor.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
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
            "Scalar multiplier for the product of input tensors A * B, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "beta",
            "Scalar multiplier for input tensor C, the default value is 1.0.",
            AttributeProto::FLOAT,
            1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA =
                transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB =
                transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;

            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(transA ? 1 : 0);
            *ctx.getOutputType(0)
                 ->mutable_tensor_type()
                 ->mutable_shape()
                 ->add_dim() =
                ctx.getInputType(1)->tensor_type().shape().dim(transB ? 0 : 1);
          } else if (
              hasInputShape(ctx, 2) &&
              (!ctx.getAttribute("broadcast") ||
               static_cast<int>(ctx.getAttribute("broadcast")->i()) == 0)) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
                ctx.getInputType(2)->tensor_type().shape();
          }
        }));

static const char* Gemm_ver7_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    7,
    OpSchema()
        .SetDoc(
            Gemm_ver7_doc +
            GenerateBroadcastingDocUni("tensor C", "tensor A * B"))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T")
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T")
        .Input(
            2,
            "C",
            "Input tensor C. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T")
        .Output(0, "Y", "Output tensor of shape (M, N).", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
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
            "alpha",
            "Scalar multiplier for the product of input tensors A * B.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "beta",
            "Scalar multiplier for input tensor C.",
            AttributeProto::FLOAT,
            1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA =
                transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB =
                transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 1);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(
                ctx,
                0,
                {first_input_shape.dim(transA ? 1 : 0),
                 second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

static const char* Gemm_ver9_doc = R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    9,
    OpSchema()
        .SetDoc(
            Gemm_ver9_doc +
            GenerateBroadcastingDocUni("tensor C", "tensor A * B"))
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M, K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T")
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T")
        .Input(
            2,
            "C",
            "Input tensor C. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T")
        .Output(0, "Y", "Output tensor of shape (M, N).", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
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
            "alpha",
            "Scalar multiplier for the product of input tensors A * B.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "beta",
            "Scalar multiplier for input tensor C.",
            AttributeProto::FLOAT,
            1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2)) {
            auto transAAttr = ctx.getAttribute("transA");
            bool transA =
                transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
            auto transBAttr = ctx.getAttribute("transB");
            bool transB =
                transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
            auto& first_input_shape = getInputShape(ctx, 0);
            auto& second_input_shape = getInputShape(ctx, 1);
            if (first_input_shape.dim_size() != 2) {
              fail_shape_inference("First input does not have rank 2");
            }
            if (second_input_shape.dim_size() != 2) {
              fail_shape_inference("Second input does not have rank 2");
            }
            updateOutputShape(
                ctx,
                0,
                {first_input_shape.dim(transA ? 1 : 0),
                 second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

static const char* Max_ver6_doc = R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    6,
    OpSchema()
        .SetDoc(Max_ver6_doc)
        .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
        .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Min_ver6_doc = R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    6,
    OpSchema()
        .SetDoc(Min_ver6_doc)
        .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
        .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sum_ver6_doc = R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    6,
    OpSchema()
        .SetDoc(Sum_ver6_doc)
        .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
        .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Mean_ver6_doc = R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    6,
    OpSchema()
        .SetDoc(Mean_ver6_doc)
        .Input(
            0,
            "data_0",
            "List of tensors for Mean.",
            "T",
            OpSchema::Variadic)
        .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* MatMul_ver1_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    1,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T")
        .Input(1, "B", "N-dimensional matrix B", "T")
        .Output(0, "Y", "Matrix multiply results from A * B", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetDoc(MatMul_ver1_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }

          const auto shape0 = ctx.getInputType(0)->tensor_type().shape();
          const auto shape1 = ctx.getInputType(1)->tensor_type().shape();

          if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
            fail_shape_inference("Input tensors of wrong rank (0).");
          }

          TensorShapeProto shapeL, shapeR;

          // First promote each shape to at least rank-2. This logic is
          // specific to matmul, not generic broadcasting.
          {
            if (shape0.dim_size() == 1) {
              shapeL.add_dim()->set_dim_value(1);
              *shapeL.add_dim() = shape0.dim(0);
            } else {
              *shapeL.mutable_dim() = shape0.dim();
            }
            if (shape1.dim_size() == 1) {
              *shapeR.add_dim() = shape1.dim(0);
              shapeR.add_dim()->set_dim_value(1);
            } else {
              *shapeR.mutable_dim() = shape1.dim();
            }
          }

          // Check for compatible matrix multiply dimensions
          {
            auto dimL = shapeL.dim(shapeL.dim_size() - 1);
            auto dimR = shapeR.dim(shapeR.dim_size() - 2);
            if (dimL.has_dim_value() && dimR.has_dim_value() &&
                dimL.dim_value() != dimR.dim_value()) {
              fail_shape_inference(
                  "Incompatible dimensions for matrix multiplication");
              ;
            }
          }

          TensorShapeProto resultShape;

          // Now call out to generic multidimensional broadcasting for
          // the broadcastable prefixes.
          {
            TensorShapeProto prefixShapeL, prefixShapeR;
            for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
              *prefixShapeL.add_dim() = shapeL.dim(i);
            }
            for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
              *prefixShapeR.add_dim() = shapeR.dim(i);
            }
            bidirectionalBroadcastShapeInference(
                prefixShapeL, prefixShapeR, resultShape);
          }

          // Back to matmul-specific. Add the trailing dimensions back in.
          {
            if (shape0.dim_size() != 1) {
              *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
            }
            if (shape1.dim_size() != 1) {
              *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
            }
          }

          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
              resultShape;
        }));

static const char* TopK_ver1_doc = R"DOC(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    1,
    OpSchema()
        .SetDoc(TopK_ver1_doc)
        .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]", "T")
        .Output(
            0,
            "Values",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing top K values from the input tensor",
            "T")
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "Constrain index tensor to int64")
        .Attr(
            "k",
            "Number of top elements to retrieve",
            AttributeProto::INT,
            true)
        .Attr(
            "axis",
            "Dimension on which to do the sort.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference:
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          updateOutputElemType(ctx, 1, TensorProto::INT64);

          // Shape inference:
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int64_t rank = input_shape.dim_size();
          int64_t axis = getAttribute(ctx, "axis", -1);
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            fail_shape_inference("Invalid value for attribute axis");
          int64_t k = getAttribute(ctx, "k", -1);
          if (k <= 0)
            fail_shape_inference("Invalid value for attribute k");
          // TODO: unclear what results should be if axis has less than k
          // elements.
          TensorShapeProto result_shape = input_shape;
          result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k);
          updateOutputShape(ctx, 0, result_shape);
          updateOutputShape(ctx, 1, result_shape);
        }));

static const char* TopK_ver10_doc = R"DOC(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
   
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    10,
    OpSchema()
        .SetDoc(TopK_ver10_doc)
        .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]", "T")
        .Input(
            1,
            "K",
            "A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve",
            "tensor(int64)")
        .Output(
            0,
            "Values",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing top K values from the input tensor",
            "T")
        .Output(
            1,
            "Indices",
            "Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] "
            "containing the corresponding input tensor indices for the top K "
            "values.",
            "I")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "Constrain index tensor to int64")
        .Attr(
            "axis",
            "Dimension on which to do the sort.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference:
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          updateOutputElemType(ctx, 1, TensorProto::INT64);
          // Shape inference:
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int64_t rank = input_shape.dim_size();
          int64_t axis = getAttribute(ctx, "axis", -1);
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            fail_shape_inference("Invalid value for attribute axis");

          const auto& axis_dim = input_shape.dim(static_cast<int>(axis));
          const auto* k = ctx.getInputData(1);

          // Infer output shape if:
          // (1) 'K' is available
          // (2) axis_dim has dim value
          // Othewise cannot reliably compute output shape as axis dim value is
          // unknown and hence cannot determine if axis dim value >= k (which
          // should be enforced)
          if (nullptr != k && axis_dim.has_dim_value()) {
            int64_t k_value = 0;
            if (k->dims_size() != 1 || k->dims(0) != 1)
              fail_shape_inference(
                  "K input must be a one-dimensional tensor of size 1.");
            if (k->data_type() == TensorProto::INT64) {
              const auto& data = ParseData<int64_t>(k);
              k_value = data[0];
            } else
              fail_shape_inference("K input must be of type int64.");

            if (axis_dim.dim_value() < k_value)
              fail_shape_inference(
                  "Axis has less than the requested k elements.");

            TensorShapeProto result_shape = input_shape;
            result_shape.mutable_dim(static_cast<int>(axis))
                ->set_dim_value(k_value);

            updateOutputShape(ctx, 0, result_shape);
            updateOutputShape(ctx, 1, result_shape);

            return;
          }

          // Infer output shapes' rank in any case
          auto* output_shape_0 = getOutputShape(ctx, 0);
          auto* output_shape_1 = getOutputShape(ctx, 1);
          for (int i = 0; i < input_shape.dim_size(); ++i) {
            output_shape_0->add_dim();
            output_shape_1->add_dim();
          }

          return;
        }));

static const char* Clip_ver6_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    6,
    OpSchema()
        .SetDoc(Clip_ver6_doc)
        .Attr(
            "min",
            "Minimum value, under which element is replaced by min",
            AttributeProto::FLOAT,
            std::numeric_limits<float>::lowest())
        .Attr(
            "max",
            "Maximum value, above which element is replaced by max",
            AttributeProto::FLOAT,
            std::numeric_limits<float>::max())
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

} // namespace ONNX_NAMESPACE
