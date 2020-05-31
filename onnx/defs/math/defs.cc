// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <functional>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Performs element-wise binary {name} (with Numpy-style broadcasting support).

{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(
            doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(0, "A", "First operand.", "T");
    schema.Input(1, "B", "Second operand.", "T");
    schema.Output(0, "C", "Result, has same element type as two inputs", "T");
    schema.TypeConstraint(
        "T",
        OpSchema::numeric_types_for_math_reduction_with_bfloat(),
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
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
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
                        ReplaceAll(doc, "{description}", description););
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
        {"tensor(float16)",
         "tensor(float)",
         "tensor(double)",
         "tensor(bfloat16)"},
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
            "'axis' must be in [",
            -r,
            " , ",
            (r - 1),
            "]. Its actual value is: ",
            axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    13,
    OpSchema().FillUsing(MathDocGenerator("addition")));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    13,
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
    13,
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
            OpSchema::all_numeric_types_with_bfloat(),
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
    13,
    OpSchema().FillUsing(MathDocGenerator("multiplication")));

ONNX_OPERATOR_SET_SCHEMA(
    Div,
    13,
    OpSchema().FillUsing(MathDocGenerator("division")));

static const char* Neg_ver13_doc = R"DOC(
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    13,
    OpSchema()
        .SetDoc(Neg_ver13_doc)
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
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Abs_ver13_doc = R"DOC(
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    13,
    OpSchema()
        .SetDoc(Abs_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_with_bfloat(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Reciprocal_ver13_doc = R"DOC(
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    13,
    OpSchema()
        .SetDoc(Reciprocal_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Floor_ver13_doc = R"DOC(
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    13,
    OpSchema()
        .SetDoc(Floor_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Ceil_ver13_doc = R"DOC(
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    13,
    OpSchema()
        .SetDoc(Ceil_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Sqrt_ver13_doc = R"DOC(
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    13,
    OpSchema()
        .SetDoc(Sqrt_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Relu_ver13_doc = R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    13,
    OpSchema()
        .SetDoc(Relu_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
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

static const char* celu_ver12_doc = R"DOC(
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula: 

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
)DOC";

static float celu_default_alpha = 1.0;

TensorProto ToDimensionOneFloatTensor(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  float alpha = ctx.getAttribute("alpha") != nullptr
      ? ctx.getAttribute("alpha")->f()
      : celu_default_alpha;
  body.push_back(
      {{"alpha"},
       "Constant",
       {},
       {MakeAttribute("value", ToDimensionOneFloatTensor(alpha))}});

  body.push_back({{"X_alpha"}, "Div", {"X", "alpha"}});
  body.push_back({{"Elu_Result"}, "Elu", {"X_alpha"}, {{"alpha", 1.f}}});
  body.push_back({{"Y"}, "Mul", {"alpha", "Elu_Result"}});

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    12,
    OpSchema()
        .SetDoc(celu_ver12_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Attr(
            "alpha",
            "The Alpha value in Celu formula which control the shape of "
            "the unit. The default value is 1.0.",
            AttributeProto::FLOAT,
            celu_default_alpha)
        .TypeConstraint(
            "T",
            {"tensor(float)"},
            "Constrain input and output types to float32 tensors.")
        .SetContextDependentFunctionBodyBuilder(
            BuildContextDependentFunctionBodyCelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Exp_ver13_doc = R"DOC(
Calculates the exponential of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    13,
    OpSchema()
        .SetDoc(Exp_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The exponential of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Log_ver13_doc = R"DOC(
Calculates the natural log of the given input tensor, element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    13,
    OpSchema()
        .SetDoc(Log_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The natural log of the input tensor computed "
            "element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Tanh_ver13_doc = R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    13,
    OpSchema()
        .SetDoc(Tanh_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Pow_ver13_doc = R"DOC(
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    13,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(
            std::string(Pow_ver13_doc) + GenerateBroadcastingDocMul()))
        .Input(0, "X", "First operand, base of the exponent.", "T")
        .Input(1, "Y", "Second operand, power of the exponent.", "T1")
        .Output(0, "Z", "Output tensor (same size as X)", "T")
        .TypeConstraint(
            "T",
            {"tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input X and output types to float/int tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)"},
            "Constrain input Y types to float/int tensors.")
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
        .SetDoc(GET_OP_DOC_STR(
            std::string(PRelu_ver9_doc) +
            GenerateBroadcastingDocUni("tensor slope", "input tensor X")))
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

static const char* Sigmoid_ver13_doc = R"DOC(
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    13,
    OpSchema()
        .SetDoc(Sigmoid_ver13_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
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

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator(
    const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
Element-wise {name} of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
{broadcast_doc}
)DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(
            doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "data_0",
        "List of tensors for " + std::string(name) + ".",
        "T",
        OpSchema::Variadic);
    schema.Output(0, name, "Output tensor.", "T");
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
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("max"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_with_bfloat(),
            "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("min"))
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_with_bfloat(),
            "Constrain input and output types to numeric tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("sum"))
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    13,
    OpSchema()
        .FillUsing(ElementwiseMultiOpDocGenerator("mean"))
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors."));

static const char* Clip_ver13_doc = R"DOC(
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    13,
    OpSchema()
        .SetDoc(Clip_ver13_doc)
        .Input(0, "input", "Input tensor whose elements to be clipped", "T")
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
            OpSchema::all_numeric_types_with_bfloat(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    13,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("softmax", "normalized exponential")));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    13,
    OpSchema().FillUsing(
        SoftmaxFamilyDocGenerator("logsoftmax", "log of softmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    13,
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

static const char* Gemm_ver13_doc = R"DOC(General Matrix multiplication:
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
    13,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(
            std::string(Gemm_ver13_doc) +
            GenerateBroadcastingDocUni("tensor C", "tensor A * B") + "\n" +
            GenerateOptionalArgumentsDoc()))
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
             "tensor(int64)",
             "tensor(bfloat16)"},
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

static const char* MatMul_ver13_doc = R"DOC(
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    13,
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
             "tensor(int64)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float/int tensors.")
        .SetDoc(MatMul_ver13_doc)
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

static const char* Expand_ver13_doc = R"DOC(
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
    13,
    OpSchema()
        .SetDoc(Expand_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Input(
            1,
            "shape",
            "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule",
            "tensor(int64)")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_with_bfloat(),
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

static const char* Sign_ver13_doc = R"DOC(
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    13,
    OpSchema()
        .SetDoc(Sign_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The sign of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_with_bfloat(),
            "Constrain input and output types to all numeric tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Erf_ver13_doc = R"DOC(
Computes the error function of the given input tensor element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    13,
    OpSchema()
        .SetDoc(Erf_ver13_doc)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The error function of the input tensor "
            "computed element-wise. It has the same shape and type of the input.",
            "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types_with_bfloat(),
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
            "Zero point tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
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
        .Input(
            1,
            "axis",
            "(Optional) A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. "
            "Negative value means counting dimensions from the back.",
            "T2")
        .Output(
            0,
            "y",
            "Output tensor of the same type as 'x' with cumulative sums of the x's elements",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float)",
             "tensor(double)"},
            "Input can be of any tensor type.")
        .TypeConstraint(
            "T2",
            {"tensor(int32)", "tensor(int64)"},
            "axis tensor can be int32 or int64 only")
        .TypeAndShapeInferenceFunction(
            ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

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
            if (mat_w.has_dim_value() && mat_h.has_dim_value() &&
                (mat_w.dim_value() != mat_h.dim_value())) {
              fail_shape_inference(
                  "The inner-most 2 dimensions must have the same size (mat_w:",
                  mat_w.dim_value(),
                  " != mat_h:",
                  mat_h.dim_value(),
                  ").");
            }

            for (int i = 0; i < rank - 2; ++i) {
              auto* dim = output_shape->add_dim();
              *dim = input_shape.dim(i);
            }
          }
        }));

static const char* NegativeLogLikelihoodLoss_ver12_doc = R"DOC(
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].

When an optional "weight" is provided, the sample loss is calculated as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].

loss is zero for the case when target-value equals ignore_index.
    
    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

    mean(loss), if "weight" is not provided,

or if weight is provided,

    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.

If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

    // negative log likelihood loss, "none" reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]

    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1]

    // print(loss)
    // [[-3. -2.]
    //  [-0. -2.]]

Example 2:

    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]

    loss = np.sum(loss)
    // print(loss)
    // -1.1

Example 3:

    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
            weight_total = weight_total + weight[c]

    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57
)DOC";

TensorProto ToDimensionOneTensor(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

TensorProto ToDimensionOneInt64Tensor(int64_t value) {
  auto t = ToTensor(std::vector<int64_t>({value}));
  t.add_dims(1);
  return t;
}

bool BuildContextDependentFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.push_back(
      {{"const_zero"},
       "Constant",
       {},
       {MakeAttribute("value", ToDimensionOneTensor(0))}});

  body.push_back(
      {{"const_one"},
       "Constant",
       {},
       {MakeAttribute("value", ToDimensionOneTensor(1))}});

  body.push_back(
      {{"expanded_target"},
       "Unsqueeze",
       {"target"},
       {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  if (ctx.getAttribute("ignore_index") == nullptr) {
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "expanded_target"},
         {MakeAttribute("axis", (int64_t)1)}});

    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element"}});

    body.push_back(
        {{"loss_N1dd"},
         "Slice",
         {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      if (ctx.getAttribute("reduction")->s() == "none") {
        body.push_back(
            {{"loss"},
             "Squeeze",
             {"loss_N1dd"},
             {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      } else {
        body.push_back(
            {{"loss_Ndd"},
             "Squeeze",
             {"loss_N1dd"},
             {MakeAttribute("axes", std::vector<int64_t>({1}))}});
        if (ctx.getAttribute("reduction")->s() == "mean") {
          body.push_back(
              {{"loss"},
               "ReduceMean",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        } else {
          body.push_back(
              {{"loss"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    } else {
      body.push_back({{"weight_gather"}, "Gather", {"weight", "target"}});
      body.push_back(
          {{"loss_unweighted"},
           "Squeeze",
           {"loss_N1dd"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      if (ctx.getAttribute("reduction")->s() == "none") {
        body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
      } else {
        body.push_back(
            {{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
        if (ctx.getAttribute("reduction")->s() == "mean") {
          body.push_back(
              {{"loss_sum"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back(
              {{"weight_gather_sum"},
               "ReduceSum",
               {"weight_gather"},
               {MakeAttribute("keepdims", (int64_t)0)}});
          body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
        } else {
          body.push_back(
              {{"loss"},
               "ReduceSum",
               {"loss_Ndd"},
               {MakeAttribute("keepdims", (int64_t)0)}});
        }
      }
    }
  } else {
    body.push_back(
        {{"const_ignore_index"},
         "Constant",
         {},
         {MakeAttribute(
             "value",
             ToDimensionOneInt64Tensor(
                 ctx.getAttribute("ignore_index")->i()))}});

    body.push_back(
        {{"const_zero_target_typed"},
         "Sub",
         {"expanded_target", "expanded_target"}});
    body.push_back(
        {{"expanded_target_int64"},
         "Cast",
         {"expanded_target"},
         {MakeAttribute(
             "to",
             (int64_t)TensorProto_DataType::TensorProto_DataType_INT64)}});

    body.push_back(
        {{"mask"}, "Equal", {"expanded_target_int64", "const_ignore_index"}});
    body.push_back(
        {{"transform_targets"},
         "Where",
         {"mask", "const_zero_target_typed", "expanded_target"}});
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "transform_targets"},
         {MakeAttribute("axis", (int64_t)1)}});
    body.push_back(
        {{"const_zero_float"},
         "Constant",
         {},
         {MakeAttribute("value", ToDimensionOneFloatTensor(0.0f))}});

    body.push_back(
        {{"input_gather_element_transform"},
         "Where",
         {"mask", "const_zero_float", "input_gather_element"}});
    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element_transform"}});
    body.push_back(
        {{"loss_N1dd"},
         "Slice",
         {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      body.push_back(
          {{"squeeze_mask"},
           "Squeeze",
           {"mask"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});

      body.push_back(
          {{"const_one_float"},
           "Constant",
           {},
           {MakeAttribute("value", ToDimensionOneFloatTensor(1.0f))}});

      body.push_back(
          {{"weight_gather"},
           "Where",
           {"squeeze_mask", "const_zero_float", "const_one_float"}});

    } else {
      body.push_back(
          {{"weight_gather_temp"}, "Gather", {"weight", "transform_targets"}});

      body.push_back(
          {{"weight_gather_temp_1"},
           "Where",
           {"mask", "const_zero_float", "weight_gather_temp"}});

      body.push_back(
          {{"weight_gather"},
           "Squeeze",
           {"weight_gather_temp_1"},
           {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    }

    body.push_back(
        {{"loss_unweighted"},
         "Squeeze",
         {"loss_N1dd"},
         {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    if (ctx.getAttribute("reduction")->s() == "none") {
      body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
    } else {
      body.push_back(
          {{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
      if (ctx.getAttribute("reduction")->s() == "mean") {
        body.push_back(
            {{"loss_sum"},
             "ReduceSum",
             {"loss_Ndd"},
             {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back(
            {{"weight_gather_sum"},
             "ReduceSum",
             {"weight_gather"},
             {MakeAttribute("keepdims", (int64_t)0)}});
        body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
      } else {
        body.push_back(
            {{"loss"},
             "ReduceSum",
             {"loss_Ndd"},
             {MakeAttribute("keepdims", (int64_t)0)}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    12,
    OpSchema()
        .SetDoc(NegativeLogLikelihoodLoss_ver12_doc)
        .Input(
            0,
            "input",
            "Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).",
            "T")
        .Input(
            1,
            "target",
            "Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the target values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind")
        .Input(
            2,
            "weight",
            "Optional rescaling weight tensor. "
            "If given, it has to be a tensor of size C. Otherwise, it is treated as if having all ones.",
            "T",
            OpSchema::Optional)
        .Output(0, "loss", "The negative log likelihood loss", "T")
        .Attr(
            "reduction",
            "Type of reduction to apply to loss: none, sum, mean (default). "
            "'none': the output is the loss for each sample. "
            "'sum': the output will be summed. "
            "'mean': the sum of the output will be divided by the sum of applied weights.",
            AttributeProto::STRING,
            std::string("mean"))
        .Attr(
            "ignore_index",
            "Specifies a target value that is ignored and does not contribute to the input gradient. It's an optional value.",
            AttributeProto::INT,
            false)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input, weight, and output types to floating-point tensors.")
        .TypeConstraint(
            "Tind",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain target to integer types")
        .SetContextDependentFunctionBodyBuilder(
            BuildContextDependentFunctionBody)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
            const TensorShapeProto& input_shape =
                ctx.getInputType(0)->tensor_type().shape();
            const TensorShapeProto& target_shape =
                ctx.getInputType(1)->tensor_type().shape();

            const int input_rank = static_cast<int>(input_shape.dim_size());
            const int target_rank = static_cast<int>(target_shape.dim_size());

            if (input_rank < 2) {
              fail_shape_inference("Input rank must be >= 2.")
            }
            if (target_rank != input_rank - 1) {
              fail_shape_inference(
                  "Target rank must be 1 less than the input rank.")
            }

            // match input dimensions (N, C, d1, ..., dk) with target
            // dimensions of (C, d1, ..., dk)
            for (int dim = 0; dim < target_rank; dim++) {
              const auto input_dim =
                  dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
              const auto target_dim = target_shape.dim(dim);
              if (input_dim.has_dim_value() && target_dim.has_dim_value() &&
                  input_dim.dim_value() != target_dim.dim_value())
                fail_shape_inference(
                    "Input and target dimension value mismatch.")
            }

            if (ctx.getNumInputs() == 3) {
              const TensorShapeProto& weight_shape =
                  ctx.getInputType(2)->tensor_type().shape();
              if (weight_shape.dim_size() != 1)
                fail_shape_inference("Weight rank must be 1.")
            }

            TensorShapeProto* output_shape =
                ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

            if (ctx.getAttribute("reduction")->s() == "none") {
              // output tensor is of shape (N, d1, d2, ..., dk) if
              // reduction attribute is "none".
              for (int i = 0; i < input_rank - 1; i++) {
                auto* dim = output_shape->add_dim();
                if (i == 0)
                  *dim = input_shape.dim(i);
                else
                  *dim = input_shape.dim(i + 1);
              }
            }
            // otherwise output is a scalar.
          }
        }));

void einsumRankInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    std::string equation) {
  const size_t numInputs = ctx.getNumInputs();
  if (numInputs < 1 || !hasNInputShapes(ctx, static_cast<int>(numInputs))) {
    return;
  }

  auto* output_shape = getOutputShape(ctx, 0);
  std::string left_equation;

  equation.erase(
      std::remove(equation.begin(), equation.end(), ' '),
      equation.end()); // Remove space char
  auto mid_index = equation.find("->");
  if (mid_index != std::string::npos) {
    // Separate right and left hand sides of the equation
    left_equation = equation.substr(0, mid_index);
  } else {
    // No right hand side
    left_equation = equation;
  }

  std::string term;
  size_t num_operands = 0;
  size_t num_ellipsis = 0;
  size_t num_ellipsis_indices = 0;

  // Parse the left-hand side
  std::stringstream str(left_equation);
  while (!str.eof()) {
    std::getline(str, term, ',');
    auto ellipsis_index = term.find("...");
    if (numInputs <= num_operands) {
      fail_shape_inference(
          "Number of input tensors does not match the operands in the equation.");
    }
    size_t rank =
        ctx.getInputType(num_operands)->tensor_type().shape().dim_size();
    if (ellipsis_index != std::string::npos) {
      // If there is an ellipsis, the number of dimensions it represents
      // must be total dim - letter dimensions
      if (num_ellipsis == 0) {
        if (rank + 3 < term.size()) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
        num_ellipsis_indices = rank - term.size() + 3;
      } else { // ellipsis has been seen before. Check that if dimensions
               // are compatible
        if (num_ellipsis_indices != rank - term.size() + 3) {
          fail_shape_inference("Ellipsis represents incompatible dimensions.");
        }
      }
      num_ellipsis++;
    } else {
      if (rank != term.size()) {
        fail_shape_inference(
            "Rank of input ",
            num_operands,
            " does not match the equation indices.");
      }
    }
    num_operands++;
  }

  if (numInputs != num_operands) {
    fail_shape_inference(
        "Number of input tensors does not match the operands in the equation.");
  }

  const size_t number_of_letters = 26;
  size_t num_letter_occurrences[number_of_letters] = {0};
  // Parse the provided right-hand side
  if (mid_index != std::string::npos) {
    std::string right_equation = equation.substr(mid_index + 2);
    auto right_ellipsis_index = right_equation.find("...");
    if (right_ellipsis_index !=
        std::string::npos) { // Right-hand side contains ellipsis
      for (size_t i = 0; i < num_ellipsis_indices; ++i) {
        output_shape->add_dim();
      }
    }
    for (char c : right_equation) { // Add a dimension per each character
                                    // in right hand equation
      if (c != '.') {
        output_shape->add_dim();
      }
    }
  } else { // Infer the dimension for right-hand side
    // If there's an ellipsis, add it's corresponding dimensions
    for (size_t i = 0; i < num_ellipsis_indices; i++) {
      output_shape->add_dim();
    }
    for (size_t i = 0; i < left_equation.size();
         i++) { // Count chars that appear exactly once on left hand side
      if ((left_equation.at(i) != ',') && (left_equation.at(i) != '.')) {
        num_letter_occurrences[left_equation.at(i) - 'a']++;
      }
    }
    for (size_t index = 0; index < number_of_letters; index++) {
      if (num_letter_occurrences[index] == 1) {
        output_shape->add_dim();
      }
    }
  }
}

static const char* Einsum_ver12_doc = R"DOC(
An einsum of the form ```term1, term2 -> output-term``` produces an output tensor using the following equation

```output[output-term] = reduce-sum( input1[term1] * input2[term] )```

where the reduce-sum performs a summation over all the indices occurring in in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    12,
    OpSchema()
        .SetDoc(Einsum_ver12_doc)
        .Attr("equation", "Einsum expression string.", AttributeProto::STRING)
        .Input(0, "Inputs", "Operands", "T", OpSchema::Variadic)
        .Output(0, "Output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrain input and output types to all numerical tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string equation = getAttribute(ctx, "equation", "");
          if (equation.compare("") == 0) {
            return;
          }
          einsumRankInference(ctx, equation);
        }));

const char* reduction_doc_sce =
    "Type of reduction to apply to loss: none, sum, mean(default). "
    "'none': no reduction will be applied, "
    "'sum': the output will be summed. "
    "'mean': the sum of the output will be divided by the number of "
    "elements in the output.";

static const char* SoftmaxCrossEntropyLoss_ver13_doc =
    R"DOC(Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
)DOC";

bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;

  body.push_back(
      {{"X_Max"},
       "ReduceMax",
       {"scores"},
       {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  body.push_back({{"X_Sub"}, "Sub", {"scores", "X_Max"}});
  body.push_back({{"X_Exp"}, "Exp", {"X_Sub"}});
  body.push_back(
      {{"X_RS"},
       "ReduceSum",
       {"X_Exp"},
       {MakeAttribute("axes", std::vector<int64_t>({1}))}});
  body.push_back({{"X_Div"}, "Div", {"X_Exp", "X_RS"}});
  body.push_back({{"X_Log"}, "Log", {"X_Div"}});

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    body.push_back({{"log_prob"}, "Identity", {"X_Log"}});
  }

  std::vector<std::string> input_tensor_names{"X_Log", "labels"};
  std::vector<FunctionBodyHelper::AttributeProtoWrapper> attributes{
      MakeRefAttribute("reduction", AttributeProto::STRING)};
  // Add weights as input if needed.
  if (ctx.hasInput(2)) {
    input_tensor_names.push_back("weights");
  }

  // add ignore_index attributes if needed.
  if (ctx.getAttribute("ignore_index") != nullptr) {
    attributes.push_back(MakeRefAttribute("ignore_index", AttributeProto::INT));
  }

  body.push_back(
      {{"output"},
       "NegativeLogLikelihoodLoss",
       input_tensor_names,
       attributes});

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    13,
    OpSchema()
        .SetDoc(SoftmaxCrossEntropyLoss_ver13_doc)
        .Attr(
            "reduction",
            reduction_doc_sce,
            AttributeProto::STRING,
            std::string("mean"))
        .Attr(
            "ignore_index",
            "Specifies a target value that is ignored and does not contribute to the input gradient. It's an optional value.",
            AttributeProto::INT,
            false)
        .Input(
            0,
            "scores",
            "The predicted outputs with shape [batch_size, class_size], or "
            "[batch_size, class_size, D1, D2 , ..., Dk], where K is the number of dimensions.",
            "T")
        .Input(
            1,
            "labels",
            "The ground truth output tensor, with shape [batch_size], or "
            "[batch_size, D1, D2, ..., Dk], where K is the number of dimensions. "
            "Labels element value shall be in range of [0, C). "
            "If ignore_index is specified, it may have a value outside [0, C) and the label values should either be "
            "in the range [0, C) or have the value ignore_index.",
            "Tind")
        .Input(
            2,
            "weights",
            "A manual rescaling weight given to each class. If given, it has to "
            "be a 1D Tensor assigning weight to each of the classes. Otherwise, "
            "it is treated as if having all ones.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Weighted loss float Tensor. If reduction is 'none', this has the "
            "shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of "
            "K-dimensional loss. Otherwise, it is a scalar.",
            "T")
        .Output(
            1,
            "log_prob",
            "Log probability tensor. If the output of softmax is prob, its value is log(prob).",
            "T",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "Tind",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain target to integer types")
        .SetContextDependentFunctionBodyBuilder(
            BuildContextDependentFunctionBodySCE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string reduction = getAttribute(ctx, "reduction", "mean");
          if (reduction.compare("none") == 0) {
            if (hasInputShape(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 1, 0);
            }
          } else {
            updateOutputShape(ctx, 0, TensorShapeProto());
          }

          if (ctx.getNumOutputs() == 2) {
            propagateElemTypeFromInputToOutput(ctx, 0, 1);
            propagateShapeFromInputToOutput(ctx, 0, 1);
          }
        }));
} // namespace ONNX_NAMESPACE
