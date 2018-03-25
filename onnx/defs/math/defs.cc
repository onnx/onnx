// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {

const char* kBroadcastDoc = R"DOC(
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

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
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
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator(
    const char* name,
    const char* description) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The operator computes the {name} ({description}) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the {name} values of the corresponding input.

X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
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
        "(int) default to 1; describes the axis of the inputs when coerced "
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
        "shape as input tensor.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(Add).SinceVersion(6).FillUsing(
    MathDocGenerator("addition"));

ONNX_OPERATOR_SCHEMA(Sub).SinceVersion(6).FillUsing(
    MathDocGenerator("subtraction"));

ONNX_OPERATOR_SCHEMA(Mul).SinceVersion(6).FillUsing(
    MathDocGenerator("multiplication"));

ONNX_OPERATOR_SCHEMA(Div).SinceVersion(6).FillUsing(
    MathDocGenerator("division"));
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SCHEMA(Neg)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Abs)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Reciprocal)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Floor)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Ceil)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sqrt)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Relu)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(LeakyRelu)
    .SinceVersion(6)
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
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Selu)
    .SinceVersion(6)
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
    .SinceVersion(6)
    .Attr(
        "alpha",
        "Coefficient of ELU default to 1.0.",
        AttributeProto::FLOAT,
        1.0f)
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
    .SinceVersion(6)
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
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Log)
    .SinceVersion(6)
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
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Tanh)
    .SinceVersion(6)
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
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Pow)
    .SetDoc(R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC" + std::string(kBroadcastDoc))
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
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(PRelu)
    .SinceVersion(6)
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
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sigmoid)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(HardSigmoid)
    .SinceVersion(6)
    .Attr("alpha", "Value of alpha default to 0.2", AttributeProto::FLOAT, 0.2f)
    .Attr("beta", "Value of beta default to 0.5", AttributeProto::FLOAT, 0.5f)
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
    .SinceVersion(6)
    .SetDoc(R"DOC(
Element-wise max of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Max.", "T", OpSchema::Variadic)
    .Output(0, "max", "Output tensor. Same dimension as inputs.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Min)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Element-wise min of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Min", "T", OpSchema::Variadic)
    .Output(0, "min", "Output tensor. Same dimension as inputs.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Sum)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Sum.", "T", OpSchema::Variadic)
    .Output(0, "sum", "Output tensor. Same dimension as inputs.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Mean)
    .SinceVersion(6)
    .SetDoc(R"DOC(
Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "List of tensors for Mean.", "T", OpSchema::Variadic)
    .Output(0, "mean", "Output tensor. Same dimension as inputs.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Clip)
    .SinceVersion(6)
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
    .Input(0, "input", "Input tensor whose elements to be clipped", "T")
    .Output(0, "output", "Output tensor with clipped input elements", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Softmax).FillUsing(
    SoftmaxFamilyDocGenerator("softmax", "normalized exponential"));

ONNX_OPERATOR_SCHEMA(LogSoftmax)
    .FillUsing(SoftmaxFamilyDocGenerator("logsoftmax", "log of softmax"));

ONNX_OPERATOR_SCHEMA(Hardmax).FillUsing(SoftmaxFamilyDocGenerator(
    "hardmax",
    "1 for the first maximum value, and 0 for all others"));

ONNX_OPERATOR_SCHEMA(Softsign)
    .SetDoc(R"DOC(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)DOC")
    .Input(0, "input", "1-D input tensor", "T")
    .Output(
        0,
        "output",
        "The softsign (x/(1+|x|)) values of the input tensor computed element-wise",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Softplus)
    .SetDoc(R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor", "T")
    .Output(0, "Y", "1D input tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Gemm)
    .SinceVersion(6)
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

ONNX_OPERATOR_SCHEMA(MatMul)
    .Input(0, "A", "N-dimensional matrix A", "T")
    .Input(1, "B", "N-dimensional matrix B", "T")
    .Output(0, "Y", "Matrix multiply results from A * B", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .SetDoc(R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC");

ONNX_OPERATOR_SCHEMA(TopK)
    .SetDoc(R"DOC(
Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC")
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
        {"tensor(int64)", "tensor(int32)"},
        "Constrain index tensor to integral types")
    .Attr("k", "Number of top elements to retrieve", AttributeProto::INT, true)
    .Attr(
        "axis",
        "Dimension on which to do the sort. Default -1, which indicates the last"
        " axis",
        AttributeProto::INT,
        static_cast<int64_t>(-1));
