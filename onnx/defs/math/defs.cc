// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Performs element-wise binary {name} (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str());
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(1, "B", "Second operand.", "T");
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction(),
        "Constrain input and output types to high-precision numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
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
    Add,
    7,
    OpSchema().FillUsing(MathDocGenerator("addition")));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    7,
    OpSchema().FillUsing(MathDocGenerator("subtraction")));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    7,
    OpSchema().FillUsing(MathDocGenerator("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(
    Div,
    7,
    OpSchema().FillUsing(MathDocGenerator("division")));

static const char* Neg_ver6_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    6,
    OpSchema()
        .SetDoc(Neg_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(int32)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(double)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Abs_ver6_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    6,
    OpSchema()
        .SetDoc(Abs_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Reciprocal_ver6_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    6,
    OpSchema()
        .SetDoc(Reciprocal_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Floor_ver6_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    6,
    OpSchema()
        .SetDoc(Floor_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Ceil_ver6_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    6,
    OpSchema()
        .SetDoc(Ceil_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sqrt_ver6_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    6,
    OpSchema()
        .SetDoc(Sqrt_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Relu_ver6_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    6,
    OpSchema()
        .SetDoc(Relu_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* LeakyRelu_ver6_doc = R"DOC(
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    6,
    OpSchema()
        .Attr("alpha", "Coefficient of leakage.", AttributeProto::FLOAT, 0.01f)
        .SetDoc(LeakyRelu_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Selu_ver6_doc = R"DOC(
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    6,
    OpSchema()
        .Attr(
            "alpha",
            "Coefficient of SELU default to 1.67326319217681884765625 "
            "(i.e., float32 approximation of 1.6732632423543772848170429916717).",
            AttributeProto::FLOAT,
            1.67326319217681884765625f)
        .Attr(
            "gamma",
            "Coefficient of SELU default to 1.05070102214813232421875 "
            "(i.e., float32 approximation of 1.0507009873554804934193349852946).",
            AttributeProto::FLOAT,
            1.05070102214813232421875f)
        .SetDoc(Selu_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Elu_ver6_doc = R"DOC(
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    6,
    OpSchema()
        .Attr("alpha", "Coefficient of ELU.", AttributeProto::FLOAT, 1.0f)
        .SetDoc(Elu_ver6_doc)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Exp_ver6_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    6,
    OpSchema()
        .SetDoc(Exp_ver6_doc)
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
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Log_ver6_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    6,
    OpSchema()
        .SetDoc(Log_ver6_doc)
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
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tanh_ver6_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    6,
    OpSchema()
        .SetDoc(Tanh_ver6_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Pow_ver7_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    7,
    OpSchema()
        .SetDoc(std::string(Pow_ver7_doc) + GenerateBroadcastingDocMul())
        .Input(0, "X", "First operand, base of the exponent.", "T")
        .Input(1, "Y", "Second operand, power of the exponent.", "T")
        .Output(0, "Z", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

static const char* PRelu_ver9_doc = R"DOC(
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    9,
    OpSchema()
        .SetDoc(
            PRelu_ver9_doc +
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
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)"},
            "Constrain input and output types to float/int tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sigmoid_ver6_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    6,
    OpSchema()
        .SetDoc(Sigmoid_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* HardSigmoid_ver6_doc = R"DOC(
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    6,
    OpSchema()
        .Attr("alpha", "Value of alpha.", AttributeProto::FLOAT, 0.2f)
        .Attr("beta", "Value of beta.", AttributeProto::FLOAT, 0.5f)
        .SetDoc(HardSigmoid_ver6_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator(
    const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
Element-wise {name} of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
{broadcast_doc}
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str());
    schema.SetDoc(doc);
    schema.Input(
        0,
        "data_0",
        "List of tensors for " + std::string(name) + ".",
        "T",
        OpSchema::Variadic);
    schema.Output(0, name, "Output tensor.", "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const TensorShapeProto*> shapes;
      for (int i = 0; i < num_inputs; ++i) {
        auto input_type = ctx.getInputType(i);
        if (nullptr == input_type || !input_type->has_tensor_type() ||
            !input_type->tensor_type().has_shape()) {
          return;
        }
        shapes.push_back(&input_type->tensor_type().shape());
      }

      multidirectionalBroadcastShapeInference(
          shapes,
          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    8,
    OpSchema().FillUsing(ElementwiseMultiOpDocGenerator("max")));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    8,
    OpSchema().FillUsing(ElementwiseMultiOpDocGenerator("min")));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    8,
    OpSchema().FillUsing(ElementwiseMultiOpDocGenerator("sum")));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    8,
    OpSchema().FillUsing(ElementwiseMultiOpDocGenerator("mean")));

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

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    1,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    1,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    1,
    OpSchema().FillUsing(SoftmaxFamilyDocGenerator(
        "hardmax",
        "1 for the first maximum value, and 0 for all others")));

static const char* Softsign_ver1_doc = R"DOC(
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    1,
    OpSchema()
        .SetDoc(Softsign_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The softsign (x/(1+|x|)) values of the input tensor computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Softplus_ver1_doc = R"DOC(
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    1,
    OpSchema()
        .SetDoc(Softplus_ver1_doc)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

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
            if (first_input_shape.dim_size() != 2)
              fail_shape_inference("First input does not have rank 2");
            if (second_input_shape.dim_size() != 2)
              fail_shape_inference("Second input does not have rank 2");
            updateOutputShape(
                ctx,
                0,
                {first_input_shape.dim(transA ? 1 : 0),
                 second_input_shape.dim(transB ? 0 : 1)});
          }
        }));

static const char* MatMul_ver9_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    9,
    OpSchema()
        .Input(0, "A", "N-dimensional matrix A", "T")
        .Input(1, "B", "N-dimensional matrix B", "T")
        .Output(0, "Y", "Matrix multiply results from A * B", "T")
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
        .SetDoc(MatMul_ver9_doc)
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

static const char* Sin_ver7_doc = R"DOC(
Calculates the sine of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    7,
    OpSchema()
        .SetDoc(Sin_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The sine of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Cos_ver7_doc = R"DOC(
Calculates the cosine of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    7,
    OpSchema()
        .SetDoc(Cos_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The cosine of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tan_ver7_doc = R"DOC(
Calculates the tangent of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    7,
    OpSchema()
        .SetDoc(Tan_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The tangent of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Asin_ver7_doc = R"DOC(
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    7,
    OpSchema()
        .SetDoc(Asin_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The arcsine of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Acos_ver7_doc = R"DOC(
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    7,
    OpSchema()
        .SetDoc(Acos_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The arccosine of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Atan_ver7_doc = R"DOC(
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    7,
    OpSchema()
        .SetDoc(Atan_ver7_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The arctangent of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Expand_ver8_doc = R"DOC(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    8,
    OpSchema()
        .SetDoc(Expand_ver8_doc)
        .Input(0, "input", "Input tensor", "T")
        .Input(
            1,
            "shape",
            "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
            "tensor(int64)")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensors."));

static const char* Sinh_ver9_doc = R"DOC(
Calculates the hyperbolic sine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    9,
    OpSchema()
        .SetDoc(Sinh_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic sine values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Cosh_ver9_doc = R"DOC(
Calculates the hyperbolic cosine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    9,
    OpSchema()
        .SetDoc(Cosh_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic cosine values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Asinh_ver9_doc = R"DOC(
Calculates the hyperbolic arcsine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    9,
    OpSchema()
        .SetDoc(Asinh_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic arcsine values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Acosh_ver9_doc = R"DOC(
Calculates the hyperbolic arccosine of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    9,
    OpSchema()
        .SetDoc(Acosh_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic arccosine values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Atanh_ver9_doc = R"DOC(
Calculates the hyperbolic arctangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    9,
    OpSchema()
        .SetDoc(Atanh_ver9_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic arctangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));


static const char* Sign_ver9_doc = R"DOC(
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
	Sign,
	9,
	OpSchema()
		.SetDoc(Sign_ver9_doc)
		.Input(0, "input", "Input tensor", "T")
		.Output(
			0,
			"output",
			"The sign of the input tensor "
			"computed element-wise. It has the same shape and type of the input.",
			"T")
		.TypeConstraint(
			"T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Erf_ver9_doc = R"DOC(
Computes the error function of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
	Erf,
	9,
	OpSchema()
		.SetDoc(Erf_ver9_doc)
		.Input(0, "input", "Input tensor", "T")
		.Output(
			0,
			"output",
			"The error function of the input tensor "
			"computed element-wise. It has the same shape and type of the input.",
			"T")
		.TypeConstraint(
			"T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
} // namespace ONNX_NAMESPACE
