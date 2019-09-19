// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <functional>
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

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
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the {name} values of the corresponding input.
)DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{description}", description);
    schema.SetDoc(doc);
    schema.Attr(
        "axis",
        "Describes the axis of the inputs when coerced "
        "to 2D; defaults to one because the 0th axis most likely describes "
        "the batch_size. Negative value means counting dimensions "
        "from the back. Accepted range is [-r, r-1] where r = rank(input).",
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
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      
      // Shape inference starts
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      // Validate the value of 'axis'
      const TensorShapeProto& input_shape =
        ctx.getInputType(0)->tensor_type().shape();
      int r = input_shape.dim_size();
      int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
      if (axis < -r || axis >= r) {
          fail_shape_inference(
         "'axis' must be in [" -r, " , " , (r-1) , "]. Its actual value is: ", axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
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

static const char* Mod_doc = R"DOC(
  Performs element-wise binary modulus (with Numpy-style broadcasting support). 
    The sign of the remainder is the same as that of the Divisor.
  
    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend 
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
    This attribute is set to 0 by default causing the behavior to be like integer mod. 
    Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

    If the input type is floating point, then `fmod` attribute must be set to 1.
  
    In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    10,
    OpSchema()
        .SetDoc(Mod_doc)
        .Attr(
            "fmod",
            "Whether the operator should behave like fmod (default=0 meaning it will do integer mods); Set this to 1 to force fmod treatment",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "A", "Dividend tensor", "T")
        .Input(1, "B", "Divisor tensor", "T")
        .Output(0, "C", "Remainder tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to high-precision numeric tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 2))
            bidirectionalBroadcastShapeInference(
                ctx.getInputType(0)->tensor_type().shape(),
                ctx.getInputType(1)->tensor_type().shape(),
                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
        }));

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

static const char* ThresholdedRelu_ver10_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    10,
    OpSchema()
        .SetDoc(ThresholdedRelu_ver10_doc)
        .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
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

static const char* Clip_ver11_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    11,
    OpSchema()
        .SetDoc(Clip_ver11_doc)
        .Input(
            0,
            "input",
            "Input tensor whose elements to be clipped",
            "T")
        .Input(
            1,
            "min",
            "Minimum value, under which element is replaced by min. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Input(
            2,
            "max",
            "Maximum value, above which element is replaced by max. "
            "It must be a scalar(tensor of empty shape).",
            "T",
            OpSchema::Optional)
        .Output(0, "output", "Output tensor with clipped input elements", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    11,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    11,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    11,
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

static const char* Gemm_ver11_doc = R"DOC(General Matrix multiplication:
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
    11,
    OpSchema()
        .SetDoc(
            Gemm_ver11_doc +
            GenerateBroadcastingDocUni("tensor C", "tensor A * B") +
            "\n" +
            GenerateOptionalArgumentsDoc())
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
            "Optional input tensor C. "
            "If not specified, the computation is done as if C is a scalar 0. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T",
            OpSchema::Optional)
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

void matmulShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

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
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
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

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

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
          matmulShapeInference(ctx, 0, 1);
        }));

static const char* TopK_ver11_doc = R"DOC(
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

If "largest" is 1 (the default value) then the k largest elements are returned.
If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
 a tiebreaker. That is, the element with the lower index will appear first.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    11,
    OpSchema()
        .SetDoc(TopK_ver11_doc)
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
            OpSchema::all_numeric_types(),
            "Constrain input and output types to numeric tensors.")
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "Constrain index tensor to int64")
        .Attr(
            "axis",
            "Dimension on which to do the sort. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "largest",
            "Whether to return the top-K largest or smallest elements.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "sorted",
            "Whether to return the elements in sorted order.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
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
            "Constrain input and output types to all tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          // For shape inference (and rank inference), we need both input shape
          // and values in 'shape' tensor
          const auto* shape_initializer = ctx.getInputData(1);
          if (hasNInputShapes(ctx, 1) && nullptr != shape_initializer) {
            const auto& shape_initializer_shape =
                ctx.getInputType(1)->tensor_type().shape();
            if (shape_initializer_shape.dim_size() != 1 ||
                shape_initializer->data_type() != TensorProto::INT64)
              fail_shape_inference(
                  "'shape' input must be 1D tensor of type INT64");

            const auto& input_shape =
                ctx.getInputType(0)->tensor_type().shape();
            const auto& shape_data = ParseData<int64_t>(shape_initializer);

            TensorShapeProto second_shape;
            for (const auto& e : shape_data) {
              auto* dim = second_shape.add_dim();
              dim->set_dim_value(e);
            }

            bidirectionalBroadcastShapeInference(
                input_shape, second_shape, *getOutputShape(ctx, 0));
          }
          return;
        }));

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

static const char* QLinearMatMul_ver10_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output.
The quantization formula is y = saturate((x / y_scale) + y_zero_point). For (x / y_scale), it is rounding to nearest ties to even.
Refer to https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point must have same shape.
They must be either scalar (per tensor) or 1-D tensor (per row for 'a' and per column for 'b'). If scale and zero point are 1-D tensor,
the number of elements of scale and zero point tensor of input 'a' and output 'y' should be equal to the number of rows of input 'a',
and the number of elements of scale and zero point tensor of input 'b' should be equal to the number of columns of input 'b'.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    10,
    OpSchema()
        .SetDoc(QLinearMatMul_ver10_doc)
        .Input(0, "a", "N-dimensional quantized matrix a", "T1")
        .Input(1, "a_scale", "scale of quantized input a", "tensor(float)")
        .Input(2, "a_zero_point", "zero point of quantized input a", "T1")
        .Input(3, "b", "N-dimensional quantized matrix b", "T2")
        .Input(4, "b_scale", "scale of quantized input b", "tensor(float)")
        .Input(5, "b_zero_point", "zero point of quantized input b", "T2")
        .Input(6, "y_scale", "scale of quantized output y", "tensor(float)")
        .Input(7, "y_zero_point", "zero point of quantized output y", "T3")
        .Output(0, "y", "Quantized matrix multiply results from a * b", "T3")
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input a and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input b and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain output y and its zero point data type to 8-bit integer tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext&
                                              ctx) {
          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(3);
          if (nullptr == a_type || nullptr == b_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          auto a_zero_point_type = ctx.getInputType(2);
          if (nullptr == a_zero_point_type ||
              a_zero_point_type->tensor_type().elem_type() !=
                  a_type->tensor_type().elem_type()) {
            fail_type_inference(
                "input and zero_point pair is expected to have be same type.");
          }

          auto b_zero_point_type = ctx.getInputType(5);
          if (nullptr == b_zero_point_type ||
              b_zero_point_type->tensor_type().elem_type() !=
                  b_type->tensor_type().elem_type()) {
            fail_type_inference(
                "input and zero_point pair is expected to have same type.");
          }

          propagateElemTypeFromInputToOutput(ctx, 7, 0);

          matmulShapeInference(ctx, 0, 3);
        }));

static const char* MatMulInteger_ver10_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMulInteger,
    10,
    OpSchema()
        .SetDoc(MatMulInteger_ver10_doc)
        .Input(0, "A", "N-dimensional matrix A", "T1")
        .Input(1, "B", "N-dimensional matrix B", "T2")
        .Input(
            2,
            "a_zero_point",
            "Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-row quantization. If it's a 1-D tensor, its number of elements "
            "should be equal to the number of rows of input 'A'.",
            "T1",
            OpSchema::Optional)
        .Input(
            3,
            "b_zero_point",
            "Scale tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
            "which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of columns of input 'B'.",
            "T2",
            OpSchema::Optional)
        .Output(0, "Y", "Matrix multiply results from A * B", "T3")
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input A data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input B data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T3",
            {"tensor(int32)"},
            "Constrain output Y data type as 32-bit integer tensor.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext&
                                              ctx) {
          auto a_type = ctx.getInputType(0);
          auto b_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
              a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
              b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
            fail_type_inference(
                "inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(
              ONNX_NAMESPACE::TensorProto::INT32);

          matmulShapeInference(ctx, 0, 1);
        }));
static const char* CumSum_ver11_doc = R"DOC(
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
 )DOC";

ONNX_OPERATOR_SET_SCHEMA(
      CumSum,
      11,
    OpSchema()
            .SetDoc(CumSum_ver11_doc)
      .Attr(
          "exclusive",
          "If set to 1 will return exclusive sum in which the top element is not included."
          " In other terms, if set to 1, the j-th output element would be the sum of the first (j-1) elements."
          " Otherwise, it would be the sum of the first j elements.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Attr(
          "reverse",
          "If set to 1 will perform the sums in reverse direction.",
          AttributeProto::INT,
          static_cast<int64_t>(0))
      .Input(0, "x", "An input tensor that is to be processed.", "T")
      .Input(1, "axis", "(Optional) A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. "
            "Negative value means counting dimensions from the back.", "T2")
      .Output(0, "y",
              "Output tensor of the same type as 'x' with cumulative sums of the x's elements",
              "T")
      .TypeConstraint("T",  {
        "tensor(uint32)",
        "tensor(uint64)",
        "tensor(int32)",
        "tensor(int64)",
        "tensor(float)",
        "tensor(double)"}, "Input can be of any tensor type.")
      .TypeConstraint("T2", {
        "tensor(int32)",
        "tensor(int64)"}, "axis tensor can be int32 or int64 only")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      );

static const char* Round_ver11_doc = R"DOC(
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halfs, the rule is to round them to the nearest even integer.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    11,
    OpSchema()
        .SetDoc(Round_ver11_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Det_ver11_doc = R"DOC(
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    11,
    OpSchema()
        .SetDoc(Det_ver11_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to floating-point tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& input_shape =
                ctx.getInputType(0)->tensor_type().shape();
            TensorShapeProto* output_shape =
                ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            const int rank = static_cast<int>(input_shape.dim_size());

            if (rank < 2) {
              fail_shape_inference("Input rank must be >= 2.")
            }

            const auto mat_w = input_shape.dim(rank - 1);
            const auto mat_h = input_shape.dim(rank - 2);
            if (mat_w.has_dim_value() &&
                mat_h.has_dim_value() &&
                (mat_w.dim_value() != mat_h.dim_value())) {
              fail_shape_inference(
                  "The inner-most 2 dimensions must have the same size (mat_w:",
                  mat_w.dim_value(),
                  " != mat_h:",
                  mat_h.dim_value(),
                  ").");
            }

            for (int i=0; i < rank - 2; ++i) {
              auto* dim = output_shape->add_dim();
              *dim = input_shape.dim(i);
            }
          }
        }));

} // namespace ONNX_NAMESPACE
