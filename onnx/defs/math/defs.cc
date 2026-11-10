// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void MathOpDataPropagator(DataPropagationContext& ctx, std::string_view op_type) {
  const auto* const input_0 = ctx.getInputData(0);
  const auto* const input_1 = ctx.getInputData(1);
  if (input_0 == nullptr || input_1 == nullptr) {
    return;
  }
  int size_0 = input_0->dim_size();
  int size_1 = input_1->dim_size();
  // Fails to broadcast if the ranks are different and no any rank is 1
  if (size_0 != size_1 && size_0 != 1 && size_1 != 1) {
    fail_shape_inference("Invalid rank for ", op_type, " broadcasting: (", size_0, ") vs (", size_1, ").");
  }
  TensorShapeProto tsp;
  int size_out = size_0 == 1 ? size_1 : size_0;
  for (int i = 0; i < size_out; ++i) {
    const auto& input_dim_0 = input_0->dim(size_0 == 1 ? 0 : i);
    const auto& input_dim_1 = input_1->dim(size_1 == 1 ? 0 : i);
    if (input_dim_0.has_dim_value() && input_dim_1.has_dim_value()) {
      tsp.mutable_dim()->Add()->set_dim_value(
          defs::math::utils::MathOpTwoIntegers(op_type, input_dim_0.dim_value(), input_dim_1.dim_value()));
    } else {
      // Cannot compute the value; simply add an empty dim without value and param
      tsp.mutable_dim()->Add();
    }
  }
  ctx.addOutputData(0, std::move(tsp));
}

static std::function<void(OpSchema&)> MathDocGenerator() {
  return [=](OpSchema& schema) {
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

ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,
    generated::ai_onnx::Add_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Add"); }));

ONNX_OPERATOR_SET_SCHEMA(
    Sub,
    14,
    generated::ai_onnx::Sub_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Sub"); }));

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    28,
    generated::ai_onnx::Mod_ver28().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Mul,
    14,
    generated::ai_onnx::Mul_ver14()
        .FillUsing(MathDocGenerator())
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { MathOpDataPropagator(ctx, "Mul"); }));

ONNX_OPERATOR_SET_SCHEMA(Div, 14, generated::ai_onnx::Div_ver14().FillUsing(MathDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    13,
    generated::ai_onnx::Neg_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    13,
    generated::ai_onnx::Abs_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    13,
    generated::ai_onnx::Reciprocal_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    13,
    generated::ai_onnx::Floor_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    13,
    generated::ai_onnx::Ceil_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    13,
    generated::ai_onnx::Sqrt_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    14,
    generated::ai_onnx::Relu_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    generated::ai_onnx::LeakyRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    22,
    generated::ai_onnx::ThresholdedRelu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    22,
    generated::ai_onnx::Selu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    22,
    generated::ai_onnx::Elu_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    22,
    generated::ai_onnx::Mish_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    28,
    generated::ai_onnx::Celu_ver28().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static constexpr const char* gelu_default_approx = "none";

static bool BuildContextDependentFunctionBodyGelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* const approx_attr_proto = ctx.getAttribute("approximate");
  std::string approximate =
      approx_attr_proto != nullptr && approx_attr_proto->has_s() ? approx_attr_proto->s() : gelu_default_approx;
  FunctionBuilder builder(functionProto);

  if (approximate == "tanh") {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              TwoOverPi = Constant <value = float {0.63661977236}>()
              TwoOverPiCast = CastLike (TwoOverPi, X)
              C0 = Constant <value = float {0.044715}>()
              C0Cast = CastLike (C0, X)
              SqrtTwoOverPi = Sqrt (TwoOverPiCast)
              Three = Constant <value = float {3.0}>()
              ThreeCast = CastLike (Three, X)
              XCubed = Pow (X, ThreeCast)
              XCubedC0 = Mul (C0Cast, XCubed)
              XC0XCubed = Sum (X, XCubedC0)
              TanhInput = Mul (SqrtTwoOverPi, XC0XCubed)
              ErfApprox = Tanh (TanhInput)
              PhiApprox = Sum (OneCast, ErfApprox)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, PhiApprox)
              )");
  } else {
    builder.Add(R"(
              Half = Constant <value = float {0.5}>()
              HalfCast = CastLike (Half, X)
              One = Constant <value = float {1.0}>()
              OneCast = CastLike (One, X)
              Two = Constant <value = float {2.0}>()
              TwoCast = CastLike (Two, X)
              SqrtTwo = Sqrt (TwoCast)
              XSqrt = Div (X, SqrtTwo)
              ErfXSqrt = Erf(XSqrt)
              Phi = Sum (OneCast, ErfXSqrt)
              MultX = Mul (HalfCast, X)
              Y = Mul (MultX, Phi)
              )");
  }
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Gelu,
    20,
    generated::ai_onnx::Gelu_ver20()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyGelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Swish,
    24,
    generated::ai_onnx::Swish_ver24().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void SwiGLUShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }
  // A and B must have identical shapes: no broadcasting, no rank expansion, no
  // size-1 stretching. Seed the output with A's shape and merge B into it, which
  // fails on any rank mismatch or conflicting static dimension while merging
  // symbolic/unknown dimensions toward the more-specified one.
  auto* output_type = ctx.getOutputType(0)->mutable_tensor_type();
  *output_type->mutable_shape() = ctx.getInputType(0)->tensor_type().shape();
  mergeInShapeInfo(ctx.getInputType(1)->tensor_type(), *output_type);
}

ONNX_OPERATOR_SET_SCHEMA(
    SwiGLU,
    28,
    generated::ai_onnx::SwiGLU_ver28().TypeAndShapeInferenceFunction(SwiGLUShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    13,
    generated::ai_onnx::Exp_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    13,
    generated::ai_onnx::Log_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    13,
    generated::ai_onnx::Tanh_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    15,
    generated::ai_onnx::Pow_ver15().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    16,
    generated::ai_onnx::PRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    13,
    generated::ai_onnx::Sigmoid_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    22,
    generated::ai_onnx::HardSigmoid_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    22,
    generated::ai_onnx::HardSwish_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      int num_inputs = static_cast<int>(ctx.getNumInputs());
      std::vector<const TensorShapeProto*> shapes;
      shapes.reserve(num_inputs);
      for (int i = 0; i < num_inputs; ++i) {
        const auto* const input_type = ctx.getInputType(i);
        if (nullptr == input_type || !input_type->has_tensor_type() || !input_type->tensor_type().has_shape()) {
          return;
        }
        shapes.push_back(&input_type->tensor_type().shape());
      }

      multidirectionalBroadcastShapeInference(shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Max, 13, generated::ai_onnx::Max_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Min, 13, generated::ai_onnx::Min_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Sum, 13, generated::ai_onnx::Sum_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(Mean, 13, generated::ai_onnx::Mean_ver13().FillUsing(ElementwiseMultiOpDocGenerator()));

static bool BuildContextDependentFunctionBodyClip(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  bool has_min = ctx.hasInput(1);
  bool has_max = ctx.hasInput(2);

  FunctionBuilder builder(functionProto);
  if (!has_min && !has_max) {
    builder.Add("output = Identity (input)");
  } else if (has_min && !has_max) {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("output = Where (input_less_than_min, min, input)");
  } else if (!has_min && has_max) {
    builder.Add("input_large_than_max = Less (max, input)");
    builder.Add("output = Where (input_large_than_max, max, input)");
  } else {
    builder.Add("input_less_than_min = Less (input, min)");
    builder.Add("tmp = Where (input_less_than_min, min, input)");
    builder.Add("output_large_than_max = Less (max, tmp)");
    builder.Add("output = Where (output_large_than_max, max, tmp)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    13,
    generated::ai_onnx::Clip_ver13()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyClip)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([name](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference starts
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      // Validate the value of 'axis'
      const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int r = input_shape.dim_size();
      if (r == 0) {
        fail_shape_inference("Input rank must be >= 1 for ", name, ".");
      }
      int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    13,
    generated::ai_onnx::Softmax_ver13()
        .FillUsing(SoftmaxFamilyDocGenerator("Softmax"))
        // function body builder for opset version 13 (the default opset version is the same
        // as the operator's since_version.
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            })
        // function body builder for opset version 18.
        // ReduceSum is updated in opset 18 to have axes as the second input.
        // Therefore function body for opset version 18
        // is different than the one defined using opset version 13.
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    output = Div (X_Exp, X_ReduceSum)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    13,
    generated::ai_onnx::LogSoftmax_ver13()
        .FillUsing(SoftmaxFamilyDocGenerator("LogSoftmax"))
        // Function for opset 13
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis)
                  .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                  .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            13)
        // Function for opset 18
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
              FunctionBuilder builder(functionProto);
              builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");

              schema.BuildFunction(functionProto);
              return true;
            },
            18));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    13,
    generated::ai_onnx::Hardmax_ver13().FillUsing(SoftmaxFamilyDocGenerator("Hardmax")));

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    22,
    generated::ai_onnx::Softsign_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    22,
    generated::ai_onnx::Softplus_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static void GemmShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  const bool transA = getAttribute(ctx, "transA", int64_t{0}) != 0;
  const bool transB = getAttribute(ctx, "transB", int64_t{0}) != 0;
  if (hasInputShape(ctx, 1) && getInputShape(ctx, 1).dim_size() != 2) {
    const auto& b_shape = getInputShape(ctx, 1);
    fail_shape_inference(
        "Gemm input B must have rank 2, but has rank ", b_shape.dim_size(), " and shape ", b_shape, ".");
  }
  Dim K, N;
  if (transB) {
    ctx.unifyInputShape(1, {N, K});
  } else {
    ctx.unifyInputShape(1, {K, N});
  }
  if (!hasInputShape(ctx, 0)) {
    return;
  }
  auto shape = getInputShape(ctx, 0);
  const int rank = shape.dim_size();
  if (rank == 0) {
    fail_shape_inference("Gemm input A must have rank at least 1, but has rank 0 and shape ", shape, ".");
  }
  if (transA && rank > 1) {
    shape.mutable_dim()->SwapElements(rank - 2, rank - 1);
  }
  const auto& a_k = shape.dim(rank - 1);
  if (a_k.has_dim_value() && K.has_dim_value() && a_k.dim_value() != K.dim_value()) {
    fail_shape_inference(
        "Gemm inputs have incompatible contraction dimensions: A has shape ",
        getInputShape(ctx, 0),
        " with transA=",
        transA,
        ", B has shape ",
        getInputShape(ctx, 1),
        " with transB=",
        transB,
        ". After transposition, A's last dimension is ",
        a_k.dim_value(),
        " but B's first dimension is ",
        K.dim_value(),
        "; these dimensions must match.");
  }
  *shape.mutable_dim(rank - 1) = N;
  if (hasInputShape(ctx, 2)) {
    const auto& bias = getInputShape(ctx, 2);
    if (bias.dim_size() > rank) {
      fail_shape_inference(
          "Gemm input C with shape ",
          bias,
          " must be unidirectionally broadcastable to output shape ",
          shape,
          ", but C has rank ",
          bias.dim_size(),
          " greater than output rank ",
          rank,
          ".");
    }
    for (int i = 0; i < bias.dim_size(); ++i) {
      const auto& c_dim = bias.dim(i);
      const auto& y_dim = shape.dim(rank - bias.dim_size() + i);
      if (c_dim.has_dim_value() && y_dim.has_dim_value() && c_dim.dim_value() != 1 &&
          c_dim.dim_value() != y_dim.dim_value()) {
        fail_shape_inference(
            "Gemm input C with shape ",
            bias,
            " must be unidirectionally broadcastable to output shape ",
            shape,
            ", but C dimension ",
            i,
            " has size ",
            c_dim.dim_value(),
            " instead of 1 or ",
            y_dim.dim_value(),
            " (output dimension ",
            rank - bias.dim_size() + i,
            ").");
      }
    }
  }
  updateOutputShape(ctx, 0, shape);
}

static bool BuildContextDependentFunctionBodyGemm(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* a_type = ctx.getInputType(0);
  if (a_type == nullptr) {
    return false;
  }
  const auto elem_type = a_type->tensor_type().elem_type();
  if (elem_type != TensorProto::FLOAT && elem_type != TensorProto::DOUBLE && elem_type != TensorProto::FLOAT16 &&
      elem_type != TensorProto::BFLOAT16) {
    return false;
  }
  const auto* transA = ctx.getAttribute("transA");
  const auto* transB = ctx.getAttribute("transB");
  FunctionBuilder builder(functionProto);
  if (transA != nullptr && transA->i() != 0) {
    if (!a_type->tensor_type().has_shape()) {
      return false;
    }
    const int rank = a_type->tensor_type().shape().dim_size();
    std::vector<int64_t> perm;
    perm.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      perm.push_back(i);
    }
    if (rank > 1) {
      std::swap(perm[rank - 2], perm[rank - 1]);
    }
    builder.Add("AT = Transpose (A)", "perm", perm);
  } else {
    builder.Add("AT = Identity (A)");
  }
  builder.Add(transB != nullptr && transB->i() != 0 ? "BT = Transpose <perm = [1, 0]> (B)" : "BT = Identity (B)");
  builder.Add(R"ONNX(
    Product = MatMul (AT, BT)
    Alpha = Constant <value_float: float = @alpha> ()
    AlphaCast = CastLike (Alpha, A)
    ScaledProduct = Mul (Product, AlphaCast)
  )ONNX");
  const auto* beta = ctx.getAttribute("beta");
  if (ctx.hasInput(2) && (beta == nullptr || beta->f() != 0.0f)) {
    builder.Add(R"ONNX(
      Beta = Constant <value_float: float = @beta> ()
      BetaCast = CastLike (Beta, A)
      ScaledBias = Mul (C, BetaCast)
      Y = Add (ScaledProduct, ScaledBias)
    )ONNX");
  } else {
    builder.Add("Y = Identity (ScaledProduct)");
  }
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    29,
    generated::ai_onnx::Gemm_ver29()
        .TypeAndShapeInferenceFunction(GemmShapeInference)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyGemm));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    13,
    generated::ai_onnx::MatMul_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      defs::math::utils::MatMulShapeInference(ctx, 0, 1);
    }));

static void GroupedMatMulShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  Dim M, K, G, N, S;
  ctx.unifyInputShape(0, {M, K});
  ctx.unifyInputShape(1, {G, K, N});
  ctx.unifyInputShape(2, {M, S});
  if (ctx.hasInput(3)) {
    ctx.unifyInputShape(3, {G, N});
  }
  updateOutputShape(ctx, 0, {M, S, N});
}

static bool BuildContextDependentFunctionBodyGroupedMatMul(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  builder.Const1D("index_zero", int64_t{0})
      .Const1D("index_one", int64_t{1})
      .Const1D("index_two", int64_t{2})
      .Add("input_shape = Shape (input)")
      .Add("weights_shape = Shape (weights)")
      .Add("indices_shape = Shape (group_indices)")
      .Add("M = Gather <axis = 0> (input_shape, index_zero)")
      .Add("K = Gather <axis = 0> (input_shape, index_one)")
      .Add("N = Gather <axis = 0> (weights_shape, index_two)")
      .Add("S = Gather <axis = 0> (indices_shape, index_one)")
      .Add("MS = Mul (M, S)")
      .Add("indices_flat = Reshape <allowzero = 1> (group_indices, MS)")
      .Add("selected_weights = Gather <axis = 0> (weights, indices_flat)")
      .Add("input_unsqueezed = Unsqueeze (input, index_one)")
      .Add("expanded_shape = Concat <axis = 0> (M, S, K)")
      .Add("input_expanded = Expand (input_unsqueezed, expanded_shape)")
      .Add("matmul_input_shape = Concat <axis = 0> (MS, index_one, K)")
      .Add("matmul_input = Reshape <allowzero = 1> (input_expanded, matmul_input_shape)")
      .Add("matmul_result = MatMul (matmul_input, selected_weights)")
      .Add("output_shape = Concat <axis = 0> (M, S, N)")
      .Add("result = Reshape <allowzero = 1> (matmul_result, output_shape)");

  if (ctx.hasInput(3)) {
    builder.Add("selected_bias = Gather <axis = 0> (bias, indices_flat)")
        .Add("reshaped_bias = Reshape <allowzero = 1> (selected_bias, output_shape)")
        .Add("output = Add (result, reshaped_bias)");
  } else {
    builder.Add("output = Identity (result)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    GroupedMatMul,
    29,
    generated::ai_onnx::GroupedMatMul_ver29()
        .TypeAndShapeInferenceFunction(GroupedMatMulShapeInference)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyGroupedMatMul));

ONNX_OPERATOR_SET_SCHEMA(TopK, 24, generated::ai_onnx::TopK_ver24().FillUsing(defs::math::utils::TopKOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    22,
    generated::ai_onnx::Sin_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    22,
    generated::ai_onnx::Cos_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    22,
    generated::ai_onnx::Tan_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    22,
    generated::ai_onnx::Asin_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    22,
    generated::ai_onnx::Acos_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    22,
    generated::ai_onnx::Atan_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    13,
    generated::ai_onnx::Expand_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      // For shape inference, we need both input shape
      if (hasNInputShapes(ctx, 2)) {
        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        bool found = false;
        TensorShapeProto second_shape = getShapeInput(ctx, 1, found);
        if (found) {
          bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    22,
    generated::ai_onnx::Sinh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    22,
    generated::ai_onnx::Cosh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    22,
    generated::ai_onnx::Asinh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    22,
    generated::ai_onnx::Acosh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    22,
    generated::ai_onnx::Atanh_ver22().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    13,
    generated::ai_onnx::Sign_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    13,
    generated::ai_onnx::Erf_ver13().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    21,
    generated::ai_onnx::QLinearMatMul_ver21().TypeAndShapeInferenceFunction(
        defs::math::utils::QLinearMatMulShapeInference));

ONNX_OPERATOR_SET_SCHEMA(
    MatMulInteger,
    10,
    generated::ai_onnx::MatMulInteger_ver10().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      auto a_type = ctx.getInputType(0);
      auto b_type = ctx.getInputType(1);
      auto y_type = ctx.getOutputType(0);
      if (nullptr == a_type || nullptr == b_type || nullptr == y_type ||
          a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
          b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
      }

      // Right now we only support int32
      y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);

      defs::math::utils::MatMulShapeInference(ctx, 0, 1);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    CumProd,
    26,
    generated::ai_onnx::CumProd_ver26().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    14,
    generated::ai_onnx::CumSum_ver14().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    22,
    generated::ai_onnx::Round_ver22().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    22,
    generated::ai_onnx::Det_ver22().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference: input [..., M, M] -> output [...]
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        const int rank = static_cast<int>(input_shape.dim_size());

        if (rank < 2) {
          fail_shape_inference("Input rank must be >= 2.");
        }

        // Validate the inner-most 2 dims are square
        Dim M;
        unifyDim(input_shape.dim(rank - 2), M);
        unifyDim(input_shape.dim(rank - 1), M);

        // Output is the batch prefix (all dims except the last 2)
        TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        for (int i = 0; i < rank - 2; ++i) {
          *output_shape->add_dim() = input_shape.dim(i);
        }
      }
    }));

static bool BuildContextDependentFunctionBody(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  if (ctx.getInputType(0) == nullptr) {
    // we cannot create a correct function body without knowing the input type
    return false;
  }
  auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
  bool float_input = input_type == TensorProto_DataType_FLOAT;
  const auto* const reduction_attr_proto = ctx.getAttribute("reduction");
  std::string reduction_attr =
      reduction_attr_proto != nullptr && reduction_attr_proto->has_s() ? reduction_attr_proto->s() : "mean";

  FunctionBuilder builder(functionProto);
  builder.Const1D("const_zero", static_cast<int64_t>(0))
      .Const1D("const_one", static_cast<int64_t>(1))
      .Const1D("axes", static_cast<int64_t>(1))
      .Add("expanded_target = Unsqueeze (target, axes)");

  if (ctx.getAttribute("ignore_index") == nullptr) {
    builder.Add(R"(
      input_gather_element = GatherElements <axis = 1> (input, expanded_target)
      loss_NCdd = Neg (input_gather_element)
      loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)
    )");

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        builder.Add("loss = Squeeze (loss_N1dd, axes)");
      } else {
        builder.Add("loss_Ndd = Squeeze (loss_N1dd, axes)");
        if (reduction_attr == "mean") {
          builder.Add("loss = ReduceMean <keepdims = 0> (loss_Ndd)");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    } else {
      builder.Add("weight_gather = Gather (weight, target)");
      builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
      if (reduction_attr == "none") {
        builder.Add("loss = Mul (loss_unweighted, weight_gather)");
      } else {
        builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
        if (reduction_attr == "mean") {
          builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
          )");
        } else {
          builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
        }
      }
    }
  } else {
    builder.Const1D("const_ignore_index", ctx.getAttribute("ignore_index")->i());
    builder.Add(R"(
      const_zero_target_typed = Sub (expanded_target, expanded_target)
      expanded_target_int64 = Cast <to = 7> (expanded_target)
      mask = Equal (expanded_target_int64, const_ignore_index)
      transform_targets = Where (mask, const_zero_target_typed, expanded_target)
    )");
    builder.Add("input_gather_element = GatherElements <axis = 1> (input, transform_targets)");
    builder.Const1D("const_zero_float", 0.0f);
    if (!float_input) {
      builder.Add("const_zero_casted = Cast (const_zero_float)", "to", static_cast<int64_t>(input_type))
          .Add("input_gather_element_transform = Where (mask, const_zero_casted, input_gather_element)");
    } else {
      builder.Add("input_gather_element_transform = Where (mask, const_zero_float, input_gather_element)");
    }
    builder.Add("loss_NCdd = Neg (input_gather_element_transform)");
    builder.Add("loss_N1dd = Slice (loss_NCdd, const_zero, const_one, const_one)");

    if (!ctx.hasInput(2)) {
      builder.Add("squeeze_mask = Squeeze (mask, axes)");
      builder.Const1D("const_one_float", 1.0f);
      if (!float_input) {
        builder.Add("const_one_casted = Cast (const_one_float)", "to", static_cast<int64_t>(input_type))
            .Add("weight_gather = Where (squeeze_mask, const_zero_casted, const_one_casted)");
      } else {
        builder.Add("weight_gather = Where (squeeze_mask, const_zero_float, const_one_float)");
      }

    } else {
      builder.Add("weight_gather_temp = Gather (weight, transform_targets)");
      builder.Add(
          float_input ? "weight_gather_temp_1 = Where (mask, const_zero_float, weight_gather_temp)"
                      : "weight_gather_temp_1 = Where (mask, const_zero_casted, weight_gather_temp)");
      builder.Add("weight_gather = Squeeze (weight_gather_temp_1, axes)");
    }

    builder.Add("loss_unweighted = Squeeze (loss_N1dd, axes)");
    if (reduction_attr == "none") {
      builder.Add("loss = Mul (loss_unweighted, weight_gather)");
    } else {
      builder.Add("loss_Ndd = Mul (loss_unweighted, weight_gather)");
      if (reduction_attr == "mean") {
        builder.Add(R"(
            loss_sum = ReduceSum <keepdims = 0> (loss_Ndd)
            weight_gather_sum = ReduceSum <keepdims = 0> (weight_gather)
            loss = Div (loss_sum, weight_gather_sum)
        )");
      } else {
        builder.Add("loss = ReduceSum <keepdims = 0> (loss_Ndd)");
      }
    }
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    22,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver22()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          // input: [N, C, d1, ..., dk], target: [N, d1, ..., dk], weight (optional): [C]
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }

          const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();

          const int input_rank = static_cast<int>(input_shape.dim_size());
          const int target_rank = input_rank - 1;

          if (input_rank < 2) {
            fail_shape_inference("Input rank must be >= 2. input_rank=", input_rank);
          }
          checkInputRank(ctx, 1, target_rank);

          // input: [N, C, d1, ..., dk], target: [N, d1, ..., dk]
          // Build dims array for target shape: [N, d1, ..., dk]
          Dim C;
          std::vector<Dim> target_dims(target_rank);
          // Unify target dims against both inputs
          for (int i = 0; i < target_rank; i++) {
            unifyInputDim(ctx, 1, i, target_dims[i]);
            // input[0]=N matches target[0], input[2..]=d1..dk matches target[1..]
            int input_i = (i == 0) ? 0 : i + 1;
            unifyInputDim(ctx, 0, input_i, target_dims[i]);
          }
          unifyInputDim(ctx, 0, 1, C);

          // Optional weight input: shape [C]
          if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
            ctx.unifyInputShape(2, {C});
          }

          if (getAttribute(ctx, "reduction", "mean") == "none") {
            // output tensor is of shape (N, d1, d2, ..., dk)
            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            for (int i = 0; i < target_rank; i++) {
              *output_shape->add_dim() = target_dims[i];
            }
          } else {
            // output is a scalar (empty shape)
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    28,
    generated::ai_onnx::Einsum_ver28().FillUsing(defs::math::utils::EinsumOpGenerator()));

static bool BuildContextDependentFunctionBodySCE(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  // Using stable implementation of LogSoftmax
  builder //
      .Const("Shape3D", std::vector<int64_t>({0, 0, -1})) //
      .Add(R"(
        X_NCD = Reshape (scores, Shape3D)
        X_NDC = Transpose <perm = [0, 2, 1]> (X_NCD)
        X_LogSM = LogSoftmax <axis = 2> (X_NDC)
        X_LogSM_NCD = Transpose <perm = [0, 2, 1]> (X_LogSM)
        X_shape = Shape (scores)
        X_Log = Reshape (X_LogSM_NCD, X_shape)
      )");

  // Review(mzs): Ideally we want to reuse the output from Log for sub-graph
  // output as well but looking at the graph resolve code it does not include
  // graph outputs as intermediate outputs, hence if intermediate X_log is
  // renamed as log_prob then it will be treated as graph output and will not be
  // available to NegativeLogLikelihoodLoss. May be my understanding is
  // incorrect or there is a bug in function population code in ORTbut I will
  // dig further to be 100%. In the meantime we just replicate the log.
  if (ctx.hasOutput(1)) {
    builder.Add("log_prob = Identity (X_Log)");
  }

  builder.Add(
      ctx.hasInput(2)
          ? "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels, weights)"
          : "output = NegativeLogLikelihoodLoss <reduction : string = @reduction, ignore_index : int = @ignore_index> (X_Log, labels)");

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    13,
    generated::ai_onnx::SoftmaxCrossEntropyLoss_ver13()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          std::string reduction = getAttribute(ctx, "reduction", "mean");
          if (reduction == "none") {
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

ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    20,
    generated::ai_onnx::DFT_ver20().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
      bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

      const size_t input_arg_index = 0;
      const size_t dft_length_arg_index = 1;
      const size_t axis_arg_index = 2;
      const size_t output_index = 0;

      propagateElemTypeFromInputToOutput(ctx, input_arg_index, output_index);
      if (!hasInputShape(ctx, input_arg_index)) {
        // If no shape is available for the input, skip shape inference...
        return;
      }

      auto& input_shape = getInputShape(ctx, input_arg_index);

      // The last dimension is the real and imaginary parts of the value.
      const int64_t rank = input_shape.dim_size();
      if (rank < 2) {
        fail_shape_inference("input tensor must have rank >= 2, including the complex dimension.");
      }

      // check the inputs are correct types for one-sided DFT
      if (is_onesided) {
        auto last_dim = input_shape.dim(rank - 1);
        if (inverse) {
          // Check last dimension is 2 (complex input required)
          if (last_dim.has_dim_value() && last_dim.dim_value() != 2) {
            fail_shape_inference("inverse one-sided DFT requires complex input (last dimension must be 2)");
          }
        } else {
          // Check last dimension is 1 (real input required)
          if (last_dim.has_dim_value() && last_dim.dim_value() != 1) {
            fail_shape_inference("one-sided DFT requires real input (last dimension must be 1)");
          }
        }
      }

      // In general the output shape will match the input shape exactly
      // So initialize the output shape with the input shape
      TensorShapeProto result_shape_proto = input_shape;

      // Handle when axis is an input but is not statically known
      if (ctx.hasInput(axis_arg_index) && ctx.getInputData(axis_arg_index) == nullptr) {
        // Axis is an input but is not statically known
        if (is_onesided || ctx.hasInput(dft_length_arg_index)) {
          // We don't know which axis is the signal dimension, so we cannot infer shape
          // when onesided is enabled or when dft_length_arg_index is provided
          TensorShapeProto new_shape_proto{};
          for (int i = 0; i < rank; ++i) {
            new_shape_proto.add_dim();
          }
          // Set last dimension based on operation type
          ONNX_ASSERTM(
              rank == static_cast<int64_t>(new_shape_proto.dim_size()),
              "rank should be equal to new_shape_proto.dim_size()")
          if (inverse && is_onesided) {
            // IRFFT: output is real-valued
            new_shape_proto.mutable_dim(rank - 1)->set_dim_value(1);
          } else {
            // Complex output
            new_shape_proto.mutable_dim(rank - 1)->set_dim_value(2);
          }
          updateOutputShape(ctx, output_index, new_shape_proto);
          return;
        } else {
          // Set last dimension based on operation type
          int dim_size = result_shape_proto.dim_size();
          if (inverse && is_onesided) {
            // IRFFT: output is real-valued
            result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(1);
          } else {
            // Complex output
            result_shape_proto.mutable_dim(dim_size - 1)->set_dim_value(2);
          }
          updateOutputShape(ctx, output_index, result_shape_proto);
          return;
        }
      }

      // Get the axis where the DFT will be performed
      int64_t axis;
      if (!ctx.hasInput(axis_arg_index)) {
        // axis is not an input. We use -2 by default
        axis = -2;
      } else {
        const TensorProto* axis_tensor = ctx.getInputData(axis_arg_index);
        ONNX_ASSERTM(axis_tensor != nullptr, "axis should not be nullptr at this point")
        // TODO(justinchuby): Create invariance checking functions to ensure shapes and sizes
        // to abstract the following logic out.
        if (axis_tensor->dims_size() != 0) {
          fail_shape_inference("axis input must be a scalar.");
        }
        axis = defs::math::utils::GetScalarValueFromTensor<int64_t>(axis_tensor);
      }

      // NOLINTNEXTLINE(readability-simplify-boolean-expr)
      if (!(-rank <= axis && axis != -1 && axis < rank - 1)) {
        fail_shape_inference(
            "axis attribute value ",
            axis,
            " is invalid for a tensor of rank ",
            rank,
            ". Valid values are '-rank <= axis && axis != -1 && axis < rank - 1'");
      }

      auto axis_idx = (axis >= 0 ? axis : axis + rank);

      // If dft_length is specified, then we should honor the shape.
      // Set the output dimension to match the dft_length on the axis.
      if (ctx.hasInput(dft_length_arg_index)) {
        // dft_length is provided
        const TensorProto* dft_length = ctx.getInputData(dft_length_arg_index);
        if (dft_length == nullptr) {
          // If we cannot read the dft_length, we cannot infer shape on the signal axis
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
        } else {
          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be a scalar.");
          }
          auto dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

          // For RFFT, output size on signal axis is floor(dft_length/2) + 1
          if (is_onesided && !inverse) {
            // RFFT: one-sided output
            auto half_signal_size = (dft_length_value / 2) + 1;
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
          } else {
            // Standard FFT/IFFT and IRFFT: full length
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(dft_length_value);
          }
        }
      } else if (is_onesided) {
        auto axis_dimension = result_shape_proto.dim(axis_idx);
        if (axis_dimension.has_dim_value()) {
          auto axis_dimension_value = axis_dimension.dim_value();
          if (inverse) {
            // IRFFT without explicit dft_length: cannot reliably infer full signal length
            // Default to even length: N = 2 * (input_size - 1)
            auto full_signal_size = checkedMultiply(2, checkedSubtract(axis_dimension_value, 1));
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(full_signal_size);
          } else {
            // RFFT without explicit dft_length: infer one-sided output size from input
            auto half_signal_size = (axis_dimension_value / 2) + 1;
            result_shape_proto.mutable_dim(axis_idx)->set_dim_value(half_signal_size);
          }
        } else {
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_value();
          result_shape_proto.mutable_dim(axis_idx)->clear_dim_param();
        }
      }

      // Set the last dimension based on whether output is real or complex
      if (is_onesided && inverse) {
        // IRFFT: complex input -> real output (last dim = 1)
        result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(1);
      } else {
        // All other cases: complex output (last dim = 2)
        result_shape_proto.mutable_dim(static_cast<int>(rank - 1))->set_dim_value(2);
      }

      updateOutputShape(ctx, output_index, result_shape_proto);
    }));

static std::function<void(OpSchema&)> CosineSumWindowOpDocGenerator() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      // Update the output data type to the output_datatype
      auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
      updateOutputElemType(ctx, 0, narrow<int32_t>(output_datatype));

      if (!hasInputShape(ctx, 0)) {
        // If no shape is available for the input, skip shape inference.
        return;
      }

      const auto* const size = ctx.getInputData(0);
      if (size == nullptr) {
        // Size is not available, so return early
        return;
      }

      if (size->dims_size() != 0) {
        fail_shape_inference("size input must be a scalar.");
      }

      auto size_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(size);
      if (size_value <= 0) {
        fail_shape_inference("size input must be greater than 0.");
      }

      ONNX_NAMESPACE::TensorShapeProto result_shape;
      result_shape.add_dim()->set_dim_value(size_value);
      updateOutputShape(ctx, 0, result_shape);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    HannWindow,
    17,
    generated::ai_onnx::HannWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    HammingWindow,
    17,
    generated::ai_onnx::HammingWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    BlackmanWindow,
    17,
    generated::ai_onnx::BlackmanWindow_ver17().FillUsing(CosineSumWindowOpDocGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    MelWeightMatrix,
    17,
    generated::ai_onnx::MelWeightMatrix_ver17().TypeAndShapeInferenceFunction(
        [](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto output_datatype = getAttribute(ctx, "output_datatype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
          updateOutputElemType(ctx, 0, narrow<int32_t>(output_datatype));

          if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
            return;
          }

          const auto num_mel_bins = ctx.getInputData(0);
          const auto dft_length = ctx.getInputData(1);
          if (nullptr == num_mel_bins || nullptr == dft_length) {
            return;
          }

          int64_t num_mel_bins_value = -1;
          int64_t dft_length_value = -1;
          if (num_mel_bins->dims_size() != 0) {
            fail_shape_inference("num_mel_bins input must be scalar.");
          }
          num_mel_bins_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(num_mel_bins);

          if (dft_length->dims_size() != 0) {
            fail_shape_inference("dft_length input must be scalar.");
          }
          dft_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(dft_length);

          if (num_mel_bins_value > 0 && dft_length_value > 0) {
            ONNX_NAMESPACE::TensorShapeProto result_shape;
            result_shape.add_dim()->set_dim_value(static_cast<int64_t>((dft_length_value / 2) + 1));
            result_shape.add_dim()->set_dim_value(num_mel_bins_value);
            updateOutputShape(ctx, 0, result_shape);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    STFT,
    17,
    generated::ai_onnx::STFT_ver17().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0)) {
        return;
      }

      auto& input_shape = getInputShape(ctx, 0);
      if (input_shape.dim_size() != 3) {
        fail_shape_inference("Input 0 (signal) must have rank 3.");
      }

      const auto& complex_dim = input_shape.dim(2);
      if (complex_dim.has_dim_value() && complex_dim.dim_value() != 1 && complex_dim.dim_value() != 2) {
        fail_shape_inference("The last dimension of signal must have size 1 (real) or 2 (complex).");
      }

      const int64_t onesided = getAttribute(ctx, "onesided", 1);
      if (onesided != 0 && onesided != 1) {
        fail_shape_inference("Attribute onesided must be 0 or 1.");
      }
      if (onesided == 1 && complex_dim.has_dim_value() && complex_dim.dim_value() != 1) {
        fail_shape_inference("One-sided STFT requires real input (signal's last dimension must be 1).");
      }

      const auto& signal_dim = input_shape.dim(1);
      const int64_t signal_size = signal_dim.has_dim_value() ? signal_dim.dim_value() : -1;

      // Preserve compatibility with existing models that use a single-element vector.
      if (hasInputShape(ctx, 1)) {
        const auto& frame_step_shape = getInputShape(ctx, 1);
        if (frame_step_shape.dim_size() > 1 ||
            (frame_step_shape.dim_size() == 1 && frame_step_shape.dim(0).has_dim_value() &&
             frame_step_shape.dim(0).dim_value() != 1)) {
          fail_shape_inference("Input 1 (frame_step) must be a scalar or a single-element vector.");
        }
      }

      const auto* frame_step = ctx.getInputData(1);
      int64_t frame_step_value = -1;
      if (frame_step != nullptr) {
        if (frame_step->dims_size() > 1 || (frame_step->dims_size() == 1 && frame_step->dims(0) != 1)) {
          fail_shape_inference("Input 1 (frame_step) must be a scalar or a single-element vector.");
        }
        frame_step_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_step);
        if (frame_step_value <= 0) {
          fail_shape_inference("frame_step must be greater than 0.");
        }
      }

      const bool has_window = ctx.hasInput(2);
      const auto* window_shape = has_window ? getOptionalInputShape(ctx, 2) : nullptr;
      int64_t window_length = -1;
      if (window_shape != nullptr) {
        if (window_shape->dim_size() != 1) {
          fail_shape_inference("Input 2 (window) must have rank 1.");
        }
        if (window_shape->dim(0).has_dim_value()) {
          window_length = window_shape->dim(0).dim_value();
          if (window_length <= 0) {
            fail_shape_inference("window must have a positive length.");
          }
        }
      }

      const bool has_frame_length = ctx.hasInput(3);
      if (has_frame_length && hasInputShape(ctx, 3) && getInputShape(ctx, 3).dim_size() != 0) {
        fail_shape_inference("Input 3 (frame_length) must be a scalar.");
      }

      const auto* frame_length = has_frame_length ? ctx.getInputData(3) : nullptr;
      int64_t frame_length_value = -1;
      if (frame_length != nullptr) {
        if (frame_length->dims_size() != 0) {
          fail_shape_inference("Input 3 (frame_length) must be a scalar.");
        }
        frame_length_value = defs::math::utils::GetScalarValueFromTensor<int64_t>(frame_length);
        if (frame_length_value <= 0) {
          fail_shape_inference("frame_length must be greater than 0.");
        }
        if (window_length >= 0 && window_length != frame_length_value) {
          fail_type_inference(
              "If STFT has both a window input and frame_length specified, the dimension of the window "
              "must match the frame_length specified.");
        }
      }

      const bool frame_length_defaults_to_signal = !has_window && !has_frame_length;
      int64_t dft_size = frame_length_value;
      if (dft_size < 0 && window_length >= 0) {
        dft_size = window_length;
      } else if (dft_size < 0 && frame_length_defaults_to_signal) {
        dft_size = signal_size;
      }

      // The output has the following shape: [batch_size][frames][dft_unique_bins][2]
      ONNX_NAMESPACE::TensorShapeProto result_shape_proto;
      *result_shape_proto.add_dim() = input_shape.dim(0);

      if (frame_length_defaults_to_signal) {
        result_shape_proto.add_dim()->set_dim_value(1);
      } else if (frame_step_value > 0 && signal_size >= dft_size && dft_size > 0) {
        result_shape_proto.add_dim()->set_dim_value(((signal_size - dft_size) / frame_step_value) + 1);
      } else {
        result_shape_proto.add_dim();
      }

      if (dft_size > 0) {
        result_shape_proto.add_dim()->set_dim_value(onesided == 1 ? ((dft_size / 2) + 1) : dft_size);
      } else {
        result_shape_proto.add_dim();
      }
      result_shape_proto.add_dim()->set_dim_value(2);
      updateOutputShape(ctx, 0, result_shape_proto);
    }));
} // namespace ONNX_NAMESPACE
