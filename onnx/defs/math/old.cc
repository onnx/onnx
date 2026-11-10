// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/generated/schemas.h"
#include "onnx/defs/math/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static bool BuildContextDependentFunctionBody_opset13(
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
    13,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver13()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody_opset13)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasNInputShapes(ctx, 2)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

            const int input_rank = static_cast<int>(input_shape.dim_size());
            const int target_rank = static_cast<int>(target_shape.dim_size());

            if (input_rank < 2) {
              fail_shape_inference("Input rank must be >= 2.");
            }
            if (target_rank != input_rank - 1) {
              fail_shape_inference("Target rank must be 1 less than the input rank.");
            }

            // match input dimensions (N, C, d1, ..., dk) with target
            // dimensions of (C, d1, ..., dk)
            for (int dim = 0; dim < target_rank; dim++) {
              const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
              const auto target_dim = target_shape.dim(dim);
              if (input_dim.has_dim_value() && target_dim.has_dim_value() &&
                  input_dim.dim_value() != target_dim.dim_value())
                fail_shape_inference("Input and target dimension value mismatch.");
            }

            if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
              const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
              if (weight_shape.dim_size() != 1) {
                fail_shape_inference("Weight rank must be 1.");
              }
            }

            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

            if (getAttribute(ctx, "reduction", "mean") == "none") {
              // output tensor is of shape (N, d1, d2, ..., dk) if
              // reduction attribute is "none".
              for (int i = 0; i < input_rank - 1; i++) {
                auto dim = output_shape->add_dim();
                if (i == 0)
                  *dim = input_shape.dim(i);
                else
                  *dim = input_shape.dim(i + 1);
              }
            }
            // otherwise output is a scalar.
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Det,
    11,
    generated::ai_onnx::Det_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      if (hasInputShape(ctx, 0)) {
        const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        const int rank = static_cast<int>(input_shape.dim_size());

        if (rank < 2) {
          fail_shape_inference("Input rank must be >= 2.");
        }

        const auto mat_w = input_shape.dim(rank - 1);
        const auto mat_h = input_shape.dim(rank - 2);
        if (mat_w.has_dim_value() && mat_h.has_dim_value() && (mat_w.dim_value() != mat_h.dim_value())) {
          fail_shape_inference(
              "The inner-most 2 dimensions must have the same size (mat_w:",
              mat_w.dim_value(),
              " != mat_h:",
              mat_h.dim_value(),
              ").");
        }

        for (int i = 0; i < rank - 2; ++i) {
          auto dim = output_shape->add_dim();
          *dim = input_shape.dim(i);
        }
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Round,
    11,
    generated::ai_onnx::Round_ver11().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Atanh,
    9,
    generated::ai_onnx::Atanh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acosh,
    9,
    generated::ai_onnx::Acosh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asinh,
    9,
    generated::ai_onnx::Asinh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cosh,
    9,
    generated::ai_onnx::Cosh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sinh,
    9,
    generated::ai_onnx::Sinh_ver9().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Atan,
    7,
    generated::ai_onnx::Atan_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Acos,
    7,
    generated::ai_onnx::Acos_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Asin,
    7,
    generated::ai_onnx::Asin_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Tan,
    7,
    generated::ai_onnx::Tan_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Cos,
    7,
    generated::ai_onnx::Cos_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Sin,
    7,
    generated::ai_onnx::Sin_ver7().FillUsing(defs::math::utils::UnaryFloatMathOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Softplus,
    1,
    generated::ai_onnx::Softplus_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Softsign,
    1,
    generated::ai_onnx::Softsign_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSwish,
    14,
    generated::ai_onnx::HardSwish_ver14().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    HardSigmoid,
    6,
    generated::ai_onnx::HardSigmoid_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mish,
    18,
    generated::ai_onnx::Mish_ver18().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Elu,
    6,
    generated::ai_onnx::Elu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Selu,
    6,
    generated::ai_onnx::Selu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    ThresholdedRelu,
    10,
    generated::ai_onnx::ThresholdedRelu_ver10().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> MathDocGenerator_opset13() {
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

ONNX_OPERATOR_SET_SCHEMA(Add, 13, generated::ai_onnx::Add_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 13, generated::ai_onnx::Sub_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 13, generated::ai_onnx::Mul_ver13().FillUsing(MathDocGenerator_opset13()));

ONNX_OPERATOR_SET_SCHEMA(Div, 13, generated::ai_onnx::Div_ver13().FillUsing(MathDocGenerator_opset13()));

static std::function<void(OpSchema&)> MathDocGenerator_opset_7() {
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

ONNX_OPERATOR_SET_SCHEMA(Add, 7, generated::ai_onnx::Add_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 7, generated::ai_onnx::Sub_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 7, generated::ai_onnx::Mul_ver7().FillUsing(MathDocGenerator_opset_7()));

ONNX_OPERATOR_SET_SCHEMA(Div, 7, generated::ai_onnx::Div_ver7().FillUsing(MathDocGenerator_opset_7()));

static void SoftmaxFamilyShapeInference_opset11(InferenceContext& ctx, const char* name) {
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
  int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
  if (axis < -r || axis >= r) {
    fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
  }

  // Shape inference
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

ONNX_OPERATOR_SET_SCHEMA(
    Softmax,
    11,
    generated::ai_onnx::Softmax_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      SoftmaxFamilyShapeInference_opset11(ctx, "softmax");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    11,
    generated::ai_onnx::LogSoftmax_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      SoftmaxFamilyShapeInference_opset11(ctx, "logsoftmax");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Hardmax,
    11,
    generated::ai_onnx::Hardmax_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      SoftmaxFamilyShapeInference_opset11(ctx, "hardmax");
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    10,
    generated::ai_onnx::Mod_ver10().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Neg,
    6,
    generated::ai_onnx::Neg_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Abs,
    6,
    generated::ai_onnx::Abs_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reciprocal,
    6,
    generated::ai_onnx::Reciprocal_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Floor,
    6,
    generated::ai_onnx::Floor_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Ceil,
    6,
    generated::ai_onnx::Ceil_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sqrt,
    6,
    generated::ai_onnx::Sqrt_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    6,
    generated::ai_onnx::Relu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Relu,
    13,
    generated::ai_onnx::Relu_ver13().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Exp,
    6,
    generated::ai_onnx::Exp_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Log,
    6,
    generated::ai_onnx::Log_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Tanh,
    6,
    generated::ai_onnx::Tanh_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    13,
    generated::ai_onnx::Pow_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    12,
    generated::ai_onnx::Pow_ver12().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Sigmoid,
    6,
    generated::ai_onnx::Sigmoid_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

// Generate opschema for element-wise ops. Leaves type constraint "T"
// unspecified.
static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_opset8() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto num_inputs = ctx.getNumInputs();
      std::vector<const TensorShapeProto*> shapes;
      for (size_t i = 0; i < num_inputs; ++i) {
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

ONNX_OPERATOR_SET_SCHEMA(Max, 12, generated::ai_onnx::Max_ver12().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Min, 12, generated::ai_onnx::Min_ver12().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Sum, 8, generated::ai_onnx::Sum_ver8().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(Mean, 8, generated::ai_onnx::Mean_ver8().FillUsing(ElementwiseMultiOpDocGenerator_opset8()));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    12,
    generated::ai_onnx::Clip_ver12().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    13,
    generated::ai_onnx::Gemm_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto transAAttr = ctx.getAttribute("transA");
      bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
      auto transBAttr = ctx.getAttribute("transB");
      bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
      // Gemm computes Y = alpha * A' * B' + beta * C
      // A' is A or A^T depending on transA; B' is B or B^T depending on transB
      Dim M, N, K;
      if (transA) {
        ctx.unifyInputShape(0, {K, M});
      } else {
        ctx.unifyInputShape(0, {M, K});
      }
      if (transB) {
        ctx.unifyInputShape(1, {N, K});
      } else {
        ctx.unifyInputShape(1, {K, N});
      }
      // Output shape is (M, N)
      updateOutputShape(ctx, 0, {M, N});
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    11,
    generated::ai_onnx::Gemm_ver11().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2) {
          fail_shape_inference("First input does not have rank 2");
        }
        if (second_input_shape.dim_size() != 2) {
          fail_shape_inference("Second input does not have rank 2");
        }
        updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
      }
    }));

static void matmulShapeInference_opset_9(ONNX_NAMESPACE::InferenceContext& ctx, size_t input1Idx, size_t input2Idx) {
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
    const auto& dimL = shapeL.dim(shapeL.dim_size() - 1);
    const auto& dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
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
    bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
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

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = std::move(resultShape);
}

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    9,
    generated::ai_onnx::MatMul_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      matmulShapeInference_opset_9(ctx, 0, 1);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Expand,
    8,
    generated::ai_onnx::Expand_ver8().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference
      // For shape inference, we need both input shape
      const auto shape_initializer = ctx.getInputData(1);
      if (hasNInputShapes(ctx, 2)) {
        const auto& shape_input_shape = ctx.getInputType(1)->tensor_type().shape();
        if (shape_input_shape.dim_size() != 1) {
          fail_shape_inference("'shape' input must be 1D tensor");
        }

        const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
        TensorShapeProto second_shape;
        if (nullptr != shape_initializer) {
          const auto shape_data = ParseData<int64_t>(shape_initializer);

          for (const auto& e : shape_data) {
            auto dim = second_shape.add_dim();
            dim->set_dim_value(e);
          }
        } else if (shape_input_shape.dim(0).has_dim_value()) {
          // Attempt rank inference using shape of shape input
          int64_t dim_value = shape_input_shape.dim(0).dim_value();
          for (int64_t i = 0; i < dim_value; ++i) {
            second_shape.add_dim();
          }
        } else {
          return;
        }
        bidirectionalBroadcastShapeInference(input_shape, second_shape, *getOutputShape(ctx, 0));
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Sign,
    9,
    generated::ai_onnx::Sign_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Erf,
    9,
    generated::ai_onnx::Erf_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    CumSum,
    11,
    generated::ai_onnx::CumSum_ver11().TypeAndShapeInferenceFunction(
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

static TensorProto ToDimensionOneFloatTensor_old(float value) {
  auto t = ToTensor(std::vector<float>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneTensor_old(int32_t value) {
  auto t = ToTensor(std::vector<int32_t>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneInt64Tensor_old(int64_t value) {
  auto t = ToTensor(std::vector<int64_t>({value}));
  t.add_dims(1);
  return t;
}

static TensorProto ToDimensionOneInt64Tensor_old(const std::vector<int64_t>& value) {
  auto t = ToTensor(value);
  t.add_dims(static_cast<int64_t>(value.size()));
  return t;
}

static bool BuildContextDependentFunctionBody_opset12(
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
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.reserve(23);
  body.push_back({{"const_zero"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(0))}});

  body.push_back({{"const_one"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneTensor_old(1))}});

  body.push_back({{"expanded_target"}, "Unsqueeze", {"target"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

  if (ctx.getAttribute("ignore_index") == nullptr) {
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "expanded_target"},
         {MakeAttribute("axis", static_cast<int64_t>(1))}});

    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element"}});

    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      } else {
        body.push_back({{"loss_Ndd"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
        if (reduction_attr == "mean") {
          body.push_back({{"loss"}, "ReduceMean", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        }
      }
    } else {
      body.push_back({{"weight_gather"}, "Gather", {"weight", "target"}});
      body.push_back(
          {{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
      if (reduction_attr == "none") {
        body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
      } else {
        body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
        if (reduction_attr == "mean") {
          body.push_back(
              {{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
          body.push_back(
              {{"weight_gather_sum"},
               "ReduceSum",
               {"weight_gather"},
               {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
          body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
        } else {
          body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        }
      }
    }
  } else {
    body.push_back(
        {{"const_ignore_index"},
         "Constant",
         {},
         {MakeAttribute("value", ToDimensionOneInt64Tensor_old(ctx.getAttribute("ignore_index")->i()))}});

    body.push_back({{"const_zero_target_typed"}, "Sub", {"expanded_target", "expanded_target"}});
    body.push_back(
        {{"expanded_target_int64"},
         "Cast",
         {"expanded_target"},
         {MakeAttribute("to", static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_INT64))}});

    body.push_back({{"mask"}, "Equal", {"expanded_target_int64", "const_ignore_index"}});
    body.push_back({{"transform_targets"}, "Where", {"mask", "const_zero_target_typed", "expanded_target"}});
    body.push_back(
        {{"input_gather_element"},
         "GatherElements",
         {"input", "transform_targets"},
         {MakeAttribute("axis", static_cast<int64_t>(1))}});
    body.push_back(
        {{"const_zero_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(0.0f))}});
    if (!float_input) {
      body.push_back(
          {{"const_zero_casted"},
           "Cast",
           {"const_zero_float"},
           {MakeAttribute("to", static_cast<int64_t>(input_type))}});
    }
    body.push_back(
        {{"input_gather_element_transform"},
         "Where",
         {"mask", float_input ? "const_zero_float" : "const_zero_casted", "input_gather_element"}});
    body.push_back({{"loss_NCdd"}, "Neg", {"input_gather_element_transform"}});
    body.push_back({{"loss_N1dd"}, "Slice", {"loss_NCdd", "const_zero", "const_one", "const_one"}});

    if (!ctx.hasInput(2)) {
      body.push_back({{"squeeze_mask"}, "Squeeze", {"mask"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});

      body.push_back(
          {{"const_one_float"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneFloatTensor_old(1.0f))}});
      if (!float_input) {
        body.push_back(
            {{"const_one_casted"},
             "Cast",
             {"const_one_float"},
             {MakeAttribute("to", static_cast<int64_t>(input_type))}});
      }
      body.push_back(
          {{"weight_gather"},
           "Where",
           {"squeeze_mask",
            float_input ? "const_zero_float" : "const_zero_casted",
            float_input ? "const_one_float" : "const_one_casted"}});

    } else {
      body.push_back({{"weight_gather_temp"}, "Gather", {"weight", "transform_targets"}});

      body.push_back(
          {{"weight_gather_temp_1"},
           "Where",
           {"mask", float_input ? "const_zero_float" : "const_zero_casted", "weight_gather_temp"}});

      body.push_back(
          {{"weight_gather"}, "Squeeze", {"weight_gather_temp_1"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    }

    body.push_back({{"loss_unweighted"}, "Squeeze", {"loss_N1dd"}, {MakeAttribute("axes", std::vector<int64_t>({1}))}});
    if (reduction_attr == "none") {
      body.push_back({{"loss"}, "Mul", {"loss_unweighted", "weight_gather"}});
    } else {
      body.push_back({{"loss_Ndd"}, "Mul", {"loss_unweighted", "weight_gather"}});
      if (reduction_attr == "mean") {
        body.push_back({{"loss_sum"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        body.push_back(
            {{"weight_gather_sum"},
             "ReduceSum",
             {"weight_gather"},
             {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
        body.push_back({{"loss"}, "Div", {"loss_sum", "weight_gather_sum"}});
      } else {
        body.push_back({{"loss"}, "ReduceSum", {"loss_Ndd"}, {MakeAttribute("keepdims", static_cast<int64_t>(0))}});
      }
    }
  }

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto* new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    NegativeLogLikelihoodLoss,
    12,
    generated::ai_onnx::NegativeLogLikelihoodLoss_ver12()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBody_opset12)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (hasNInputShapes(ctx, 2)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const TensorShapeProto& target_shape = ctx.getInputType(1)->tensor_type().shape();

            const int input_rank = static_cast<int>(input_shape.dim_size());
            const int target_rank = static_cast<int>(target_shape.dim_size());

            if (input_rank < 2) {
              fail_shape_inference("Input rank must be >= 2. input_rank=", input_rank);
            }
            if (target_rank != input_rank - 1) {
              fail_shape_inference(
                  "Target rank must be 1 less than the input rank. input_rank=",
                  input_rank,
                  ", target_rank=",
                  target_rank);
            }

            // match input dimensions (N, C, d1, ..., dk) with target
            // dimensions of (C, d1, ..., dk)
            for (int dim = 0; dim < target_rank; dim++) {
              const auto input_dim = dim == 0 ? input_shape.dim(dim) : input_shape.dim(dim + 1);
              const auto target_dim = target_shape.dim(dim);
              if (input_dim.has_dim_value() && target_dim.has_dim_value() &&
                  input_dim.dim_value() != target_dim.dim_value())
                fail_shape_inference(
                    "Input and target dimension value mismatch. input_dim_value=",
                    input_dim.dim_value(),
                    " target_dim_value=",
                    target_dim.dim_value());
            }

            if (ctx.getNumInputs() == 3 && hasInputShape(ctx, 2)) {
              const TensorShapeProto& weight_shape = ctx.getInputType(2)->tensor_type().shape();
              const auto weight_rank = weight_shape.dim_size();
              if (weight_rank != 1) {
                fail_shape_inference("Weight rank must be 1. weight_rank=", weight_rank);
              }
            }

            TensorShapeProto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            if (getAttribute(ctx, "reduction", "mean") == "none") {
              // output tensor is of shape (N, d1, d2, ..., dk) if
              // reduction attribute is "none".
              for (int i = 0; i < input_rank - 1; i++) {
                auto dim = output_shape->add_dim();
                if (i == 0)
                  *dim = input_shape.dim(i);
                else
                  *dim = input_shape.dim(i + 1);
              }
            }
            // otherwise output is a scalar.
          }
        }));

static bool BuildContextDependentFunctionBodySCE_opset12(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  std::vector<FunctionBodyHelper::NodeDef> body;
  body.reserve(9);

  // Using stable implementation of LogSoftmax
  body.push_back({{"Shape3D"}, "Constant", {}, {MakeAttribute("value", ToDimensionOneInt64Tensor_old({0, 0, -1}))}});
  body.push_back({{"X_NCD"}, "Reshape", {"scores", "Shape3D"}});
  body.push_back({{"X_NDC"}, "Transpose", {"X_NCD"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_LogSM"}, "LogSoftmax", {"X_NDC"}, {MakeAttribute("axis", static_cast<int64_t>(2))}});
  body.push_back({{"X_LogSM_NCD"}, "Transpose", {"X_LogSM"}, {MakeAttribute("perm", std::vector<int64_t>({0, 2, 1}))}});
  body.push_back({{"X_shape"}, "Shape", {"scores"}});
  body.push_back({{"X_Log"}, "Reshape", {"X_LogSM_NCD", "X_shape"}});

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
    input_tensor_names.emplace_back("weights");
  }

  // add ignore_index attributes if needed.
  if (ctx.getAttribute("ignore_index") != nullptr) {
    attributes.emplace_back(MakeRefAttribute("ignore_index", AttributeProto::INT));
  }

  body.push_back({{"output"}, "NegativeLogLikelihoodLoss", input_tensor_names, attributes});

  auto func_nodes = FunctionBodyHelper::BuildNodes(body);
  for (const auto& node : func_nodes) {
    auto* new_node = functionProto.add_node();
    new_node->CopyFrom(node);
  }

  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SoftmaxCrossEntropyLoss,
    12,
    generated::ai_onnx::SoftmaxCrossEntropyLoss_ver12()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySCE_opset12)
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

static std::function<void(OpSchema&)> SoftmaxFamilyDocGenerator_opset1() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput); };
}

ONNX_OPERATOR_SET_SCHEMA(Softmax, 1, generated::ai_onnx::Softmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(
    LogSoftmax,
    1,
    generated::ai_onnx::LogSoftmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

ONNX_OPERATOR_SET_SCHEMA(Hardmax, 1, generated::ai_onnx::Hardmax_ver1().FillUsing(SoftmaxFamilyDocGenerator_opset1()));

static std::function<void(OpSchema&)> MathDocGenerator_old_opset6() {
  return [=](OpSchema& schema) { schema.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput); };
}

ONNX_OPERATOR_SET_SCHEMA(Add, 1, generated::ai_onnx::Add_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sub, 1, generated::ai_onnx::Sub_ver1());

ONNX_OPERATOR_SET_SCHEMA(Mul, 1, generated::ai_onnx::Mul_ver1());

ONNX_OPERATOR_SET_SCHEMA(Div, 1, generated::ai_onnx::Div_ver1());

ONNX_OPERATOR_SET_SCHEMA(Add, 6, generated::ai_onnx::Add_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Sub, 6, generated::ai_onnx::Sub_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Mul, 6, generated::ai_onnx::Mul_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(Div, 6, generated::ai_onnx::Div_ver6().FillUsing(MathDocGenerator_old_opset6()));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    1,
    generated::ai_onnx::Pow_ver1().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Pow,
    7,
    generated::ai_onnx::Pow_ver7().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

ONNX_OPERATOR_SET_SCHEMA(Neg, 1, generated::ai_onnx::Neg_ver1());

ONNX_OPERATOR_SET_SCHEMA(Abs, 1, generated::ai_onnx::Abs_ver1());

ONNX_OPERATOR_SET_SCHEMA(Reciprocal, 1, generated::ai_onnx::Reciprocal_ver1());

ONNX_OPERATOR_SET_SCHEMA(Floor, 1, generated::ai_onnx::Floor_ver1());

ONNX_OPERATOR_SET_SCHEMA(Ceil, 1, generated::ai_onnx::Ceil_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sqrt, 1, generated::ai_onnx::Sqrt_ver1());

ONNX_OPERATOR_SET_SCHEMA(Relu, 1, generated::ai_onnx::Relu_ver1());

ONNX_OPERATOR_SET_SCHEMA(LeakyRelu, 1, generated::ai_onnx::LeakyRelu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Selu, 1, generated::ai_onnx::Selu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Elu, 1, generated::ai_onnx::Elu_ver1());

ONNX_OPERATOR_SET_SCHEMA(Exp, 1, generated::ai_onnx::Exp_ver1());

ONNX_OPERATOR_SET_SCHEMA(Log, 1, generated::ai_onnx::Log_ver1());

ONNX_OPERATOR_SET_SCHEMA(Tanh, 1, generated::ai_onnx::Tanh_ver1());

ONNX_OPERATOR_SET_SCHEMA(PRelu, 1, generated::ai_onnx::PRelu_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    6,
    generated::ai_onnx::PRelu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    7,
    generated::ai_onnx::PRelu_ver7().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(Sigmoid, 1, generated::ai_onnx::Sigmoid_ver1());

ONNX_OPERATOR_SET_SCHEMA(HardSigmoid, 1, generated::ai_onnx::HardSigmoid_ver1());

ONNX_OPERATOR_SET_SCHEMA(Max, 1, generated::ai_onnx::Max_ver1());

ONNX_OPERATOR_SET_SCHEMA(Min, 1, generated::ai_onnx::Min_ver1());

ONNX_OPERATOR_SET_SCHEMA(Sum, 1, generated::ai_onnx::Sum_ver1());

ONNX_OPERATOR_SET_SCHEMA(Mean, 1, generated::ai_onnx::Mean_ver1());

ONNX_OPERATOR_SET_SCHEMA(Clip, 1, generated::ai_onnx::Clip_ver1());

ONNX_OPERATOR_SET_SCHEMA(Gemm, 1, generated::ai_onnx::Gemm_ver1());

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    6,
    generated::ai_onnx::Gemm_ver6().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;

        checkInputRank(ctx, 0, 2);
        checkInputRank(ctx, 1, 2);

        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
            first_input_shape.dim(transA ? 1 : 0);
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
            second_input_shape.dim(transB ? 0 : 1);
      } else if (
          hasInputShape(ctx, 2) &&
          (!ctx.getAttribute("broadcast") || static_cast<int>(ctx.getAttribute("broadcast")->i()) == 0)) {
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(2)->tensor_type().shape();
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    7,
    generated::ai_onnx::Gemm_ver7().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2) {
          fail_shape_inference("First input does not have rank 2");
        }
        if (second_input_shape.dim_size() != 2) {
          fail_shape_inference("Second input does not have rank 2");
        }
        updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Gemm,
    9,
    generated::ai_onnx::Gemm_ver9().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2)) {
        auto transAAttr = ctx.getAttribute("transA");
        bool transA = transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
        auto transBAttr = ctx.getAttribute("transB");
        bool transB = transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
        auto& first_input_shape = getInputShape(ctx, 0);
        auto& second_input_shape = getInputShape(ctx, 1);
        if (first_input_shape.dim_size() != 2) {
          fail_shape_inference("First input does not have rank 2");
        }
        if (second_input_shape.dim_size() != 2) {
          fail_shape_inference("Second input does not have rank 2");
        }
        updateOutputShape(ctx, 0, {first_input_shape.dim(transA ? 1 : 0), second_input_shape.dim(transB ? 0 : 1)});
      }
    }));

ONNX_OPERATOR_SET_SCHEMA(
    Max,
    6,
    generated::ai_onnx::Max_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Min,
    6,
    generated::ai_onnx::Min_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Sum,
    6,
    generated::ai_onnx::Sum_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Mean,
    6,
    generated::ai_onnx::Mean_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    MatMul,
    1,
    generated::ai_onnx::MatMul_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        auto const& dimL = shapeL.dim(shapeL.dim_size() - 1);
        auto const& dimR = shapeR.dim(shapeR.dim_size() - 2);
        if (dimL.has_dim_value() && dimR.has_dim_value() && dimL.dim_value() != dimR.dim_value()) {
          fail_shape_inference("Incompatible dimensions for matrix multiplication");
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
        bidirectionalBroadcastShapeInference(prefixShapeL, prefixShapeR, resultShape);
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
    }));

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    1,
    generated::ai_onnx::TopK_ver1().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      if (axis < 0 || axis >= rank) {
        fail_shape_inference("Invalid value for attribute axis");
      }
      int64_t k = getAttribute(ctx, "k", -1);
      if (k <= 0) {
        fail_shape_inference("Invalid value for attribute k");
      }
      TensorShapeProto result_shape = input_shape;
      result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k);
      updateOutputShape(ctx, 0, result_shape);
      updateOutputShape(ctx, 1, result_shape);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    TopK,
    10,
    generated::ai_onnx::TopK_ver10().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
      if (axis < 0 || axis >= rank) {
        fail_shape_inference("Invalid value for attribute axis");
      }

      const auto& axis_dim = input_shape.dim(static_cast<int>(axis));
      const auto k = ctx.getInputData(1);

      int64_t k_value = 0;
      if (nullptr != k) {
        if (k->dims_size() != 1 || k->dims(0) != 1) {
          fail_shape_inference("K input must be a one-dimensional tensor of size 1.");
        }

        if (k->data_type() == TensorProto::INT64) {
          const auto data = ParseData<int64_t>(k);
          k_value = data[0];
        } else {
          fail_shape_inference("K input must be of type int64.");
        }

        if (k_value <= 0) {
          fail_shape_inference("K input must contain a single positive value.");
        }
      }

      // Infer output shape if:
      // (1) 'K' is available
      // (2) axis_dim has dim value
      // Otherwise cannot reliably compute output shape as axis dim value is
      // unknown and hence cannot determine if axis dim value >= k (which
      // should be enforced)
      if (nullptr != k && axis_dim.has_dim_value()) {
        if (axis_dim.dim_value() < k_value) {
          fail_shape_inference("Axis has less than the requested k elements.");
        }

        TensorShapeProto result_shape = input_shape;
        result_shape.mutable_dim(static_cast<int>(axis))->set_dim_value(k_value);

        updateOutputShape(ctx, 0, result_shape);
        updateOutputShape(ctx, 1, result_shape);

        return;
      }

      // Infer output shapes' rank in any case
      auto output_shape_0 = getOutputShape(ctx, 0);
      auto output_shape_1 = getOutputShape(ctx, 1);
      for (int i = 0; i < input_shape.dim_size(); ++i) {
        output_shape_0->add_dim();
        output_shape_1->add_dim();
      }

      return;
    }));

ONNX_OPERATOR_SET_SCHEMA(TopK, 11, generated::ai_onnx::TopK_ver11().FillUsing(defs::math::utils::TopKOpGenerator()));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    6,
    generated::ai_onnx::Clip_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Clip,
    11,
    generated::ai_onnx::Clip_ver11().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static std::function<void(OpSchema&)> ElementwiseMultiOpDocGenerator_old() {
  return [=](OpSchema& schema) {
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      auto num_inputs = ctx.getNumInputs();
      std::vector<const TensorShapeProto*> shapes;
      for (size_t i = 0; i < num_inputs; ++i) {
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

ONNX_OPERATOR_SET_SCHEMA(Max, 8, generated::ai_onnx::Max_ver8().FillUsing(ElementwiseMultiOpDocGenerator_old()));

ONNX_OPERATOR_SET_SCHEMA(Min, 8, generated::ai_onnx::Min_ver8().FillUsing(ElementwiseMultiOpDocGenerator_old()));

ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    6,
    generated::ai_onnx::LeakyRelu_ver6().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    PRelu,
    9,
    generated::ai_onnx::PRelu_ver9().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    DFT,
    17,
    generated::ai_onnx::DFT_ver17().TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
      bool is_onesided = static_cast<bool>(getAttribute(ctx, "onesided", 0));
      bool inverse = static_cast<bool>(getAttribute(ctx, "inverse", 0));

      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasInputShape(ctx, 0)) {
        // If no shape is available for the input, skip shape inference...
        return;
      }

      // In general the output shape will match the input shape exactly
      // So initialize the output shape with the input shape
      auto& input_shape = getInputShape(ctx, 0);
      ONNX_NAMESPACE::TensorShapeProto result_shape_proto = input_shape;

      // Get the axis where the DFT will be performed.
      auto axis = static_cast<int>(getAttribute(ctx, "axis", 1));
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
      // NOLINTNEXTLINE(readability-simplify-boolean-expr)
      if (!(-rank <= axis && axis != -1 && axis < rank - 1)) {
        fail_shape_inference(
            "axis attribute value ",
            axis,
            " is invalid for a tensor of rank ",
            rank,
            ". Valid values are '-rank <= axis && axis != -1 && axis < rank - 1'");
      }

      int axis_idx = static_cast<int>(axis >= 0 ? axis : axis + rank);

      // If dft_length is specified, then we should honor the shape.
      // Set the output dimension to match the dft_length on the axis.
      const TensorProto* dft_length = nullptr;
      if (ctx.hasInput(1)) {
        dft_length = ctx.getInputData(1);
        if (dft_length == nullptr) {
          // If we cannot read the dft_length, we cannot infer shape
          // return...
          return;
        }
      }

      if (nullptr != dft_length) {
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
      auto dim_size = static_cast<int64_t>(result_shape_proto.dim_size());
      if (is_onesided && inverse) {
        // IRFFT: complex input -> real output (last dim = 1)
        result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(1);
      } else {
        // All other cases: complex output (last dim = 2)
        result_shape_proto.mutable_dim(static_cast<int>(dim_size - 1))->set_dim_value(2);
      }

      updateOutputShape(ctx, 0, result_shape_proto);
    }));

ONNX_OPERATOR_SET_SCHEMA(
    QLinearMatMul,
    10,
    generated::ai_onnx::QLinearMatMul_ver10().TypeAndShapeInferenceFunction(
        defs::math::utils::QLinearMatMulShapeInference));

static float celu_ver12_default_alpha = 1.0;

static bool BuildContextDependentFunctionBodyCelu(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  float alpha = ctx.getAttribute("alpha") != nullptr ? ctx.getAttribute("alpha")->f() : celu_ver12_default_alpha;
  FunctionBuilder builder(functionProto);
  builder.Const("alpha", std::vector<float>{alpha}).Add(R"(
            X_alpha = Div (X, alpha)
            Elu_Result = Elu <alpha = 1.0>(X_alpha)
            Y = Mul (alpha, Elu_Result)
        )");
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    Celu,
    12,
    generated::ai_onnx::Celu_ver12()
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyCelu)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Einsum,
    12,
    generated::ai_onnx::Einsum_ver12().FillUsing(defs::math::utils::EinsumOpGenerator()));
// codeql[cpp/unused-static-variable]: The schema macro defines a debug counter used by static-opset validation.
ONNX_OPERATOR_SET_SCHEMA(
    Mod,
    13,
    generated::ai_onnx::Mod_ver13().TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    }));

} // namespace ONNX_NAMESPACE
