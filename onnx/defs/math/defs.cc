// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
#include <functional>

using AttrType = onnx::OpSchema::AttrType;

namespace onnx {

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
    schema.Attr("broadcast",
                "Pass 1 to enable broadcasting",
                AttrType::INT);
    schema.Attr("axis",
                "If set, defines the broadcast dimensions. See doc for details.",
                AttrType::INT);
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and type as A");
  };
}

OPERATOR_SCHEMA(Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("addition"));

OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("subtraction"));

OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("multiplication"));

OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}, {1, 0}})
    .FillUsing(MathDocGenerator("division"));
}  // namespace onnx

OPERATOR_SCHEMA(Neg)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Abs)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Reciprocal)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Floor)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Ceil)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Sqrt)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Relu)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowConsumed({{0, 0}})
  .SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC")
  .Input(0, "X", "Input tensor")
  .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(LeakyRelu)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("alpha",
          "Coefficient of leakage",
          AttrType::FLOAT)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Selu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .Attr("alpha",
          "Coefficient of SELU default to 1.6732.",
          AttrType::FLOAT)
    .Attr("gamma",
          "Coefficient of SELU default to 1.0507.",
           AttrType::FLOAT)
    .SetDoc(R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `f(x) = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Elu)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .Attr("alpha",
          "Coefficient of ELU default to 1.0.",
          AttrType::FLOAT)
    .SetDoc(R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

OPERATOR_SCHEMA(Exp)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Calculates the exponential of the given input tensor, element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The exponential of the input tensor computed "
        "element-wise");

OPERATOR_SCHEMA(Log)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Calculates the natural log of the given input tensor, element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "Input tensor")
    .Output(
        0,
        "output",
        "The natural log of the input tensor computed "
        "element-wise");

OPERATOR_SCHEMA(Tanh)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowConsumed({{0, 0}})
  .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
    .Input(0, "input", "1-D input tensor")
    .Output(0, "output", "The hyperbolic tangent values of the input tensor "
               "computed element-wise");

OPERATOR_SCHEMA(Pow)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor of any shape, base of the exponent.")
    .Input(1, "Y", "Input tensor of any shape broadcastable to X shape, the exponent component.")    
    .Output(0, "Z", "Output tensor (same size as X)");

OPERATOR_SCHEMA(Dot)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Apply dot product between 2 tensors. Similar to numpy implementation:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
)DOC")
    .Input(0, "X", "Input tensor of any shape")
    .Input(1, "Y", "Input tensor of any shape")
    .Output(0, "Z", "Output tensor the dot product between X and Y.");

OPERATOR_SCHEMA(PRelu)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

)DOC")
    .Input(0, "X", "Input tensor")
    .Input(
        1,
        "Slope",
        "Slope tensor. If `Slope` is of size 1, the value is shared"
        "across different channels")
    .Output(0, "Y", "Input tensor");

OPERATOR_SCHEMA(Sigmoid)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowConsumed({{0, 0}})
  .SetDoc(R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor");

OPERATOR_SCHEMA(Max)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Element-wise max of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the max will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "max", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(Min)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Element-wise min of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the max will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "max", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the sum will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "sum", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(Softmax)
  .NumInputs(1)
  .NumOutputs(1)
  .SetDoc(R"DOC(
The operator computes the softmax normalized values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax normalized values of the corresponding input.

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
)DOC")
  .Attr("axis",
        "(int) default to 1; describes the axis of the inputs when coerced "
        "to 2D; defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttrType::INT)
  .Input(0, "input",
         "The input tensor that's coerced into a 2D matrix of size (NxD) "
         "as described above.")
  .Output(0, "output", "The softmax normalized output values with the same "
          "shape as input tensor.");

OPERATOR_SCHEMA(Gemm)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has dimension (M X K), input tensor B has dimension (K X N), input tensor C and output tensor Y have dimension (M X N). Input tensor C can be used inplace as the output tensor Y. If attribute broadcast is non-zero, input tensor C will be broadcasted to match the dimension requirement. If A can be transposed before doing the computation if attribute transA is non-zero, same for B and transB.
)DOC")
    .Input(0, "A", "Input tensor A")
    .Input(1, "B", "Input tensor B")
    .Input(2, "C", "Input tensor C, can be inplace.")
    .AllowConsumed({{2, 0}})
    .Output(0, "Y", "Output tensor.")
    .Attr("transA",
          "Whether A should be transposed",
          AttrType::INT)
    .Attr("transB",
          "Whether B should be transposed",
          AttrType::INT)
    .Attr("broadcast",
          "Whether C should be broadcasted",
          AttrType::INT)
    .Attr("alpha",
          "Scalar multiplier for the product of input tensors A * B",
          AttrType::FLOAT)
    .Attr("beta",
          "Scalar multiplier for input tensor C",
          AttrType::FLOAT);
