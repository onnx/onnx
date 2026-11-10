// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

static void ValidateQuantizationBlockShape(InferenceContext& ctx) {
  const auto* block_shape = ctx.getAttribute("block_shape");
  if (block_shape == nullptr) {
    return;
  }
  if (getAttribute(ctx, "block_size", 0) != 0) {
    fail_shape_inference("block_size must be zero when block_shape is specified.");
  }
  const auto rank = block_shape->ints_size();
  for (int i = 0; i < rank; ++i) {
    if (block_shape->ints(i) <= 0) {
      fail_shape_inference("block_shape entries must be positive integers.");
    }
  }
  if (!hasInputShape(ctx, 0)) {
    TensorShapeProto output_shape;
    for (int i = 0; i < rank; ++i) {
      output_shape.add_dim();
    }
    updateOutputShape(ctx, 0, output_shape);
  }
  // Validate every available shape, even if another input's rank is unknown.
  for (size_t input = 0; input < 3; ++input) {
    if (hasInputShape(ctx, input) && getInputShape(ctx, input).dim_size() != rank) {
      fail_shape_inference(
          "block_shape must have one entry per input dimension; scale and zero-point ranks must match.");
    }
  }
  for (int i = 0; i < rank; ++i) {
    for (size_t input = 1; input < 3; ++input) {
      if (!hasInputShape(ctx, input)) {
        continue;
      }
      const auto& grid_dim = getInputShape(ctx, input).dim(i);
      if (hasInputShape(ctx, 0)) {
        const auto& dim = getInputShape(ctx, 0).dim(i);
        if (dim.has_dim_value() && grid_dim.has_dim_value()) {
          const auto block = block_shape->ints(i);
          // Avoid overflow in ceiling division for large dimensions/block sizes.
          const auto expected = (dim.dim_value() / block) + (dim.dim_value() % block != 0);
          if (grid_dim.dim_value() != expected) {
            fail_shape_inference("Scale/zero-point grid dimension must equal ceil(input dimension / block_shape).");
          }
        }
      }
    }
    if (hasInputShape(ctx, 1) && hasInputShape(ctx, 2)) {
      const auto& scale_dim = getInputShape(ctx, 1).dim(i);
      const auto& zero_dim = getInputShape(ctx, 2).dim(i);
      if (scale_dim.has_dim_value() && zero_dim.has_dim_value() && scale_dim.dim_value() != zero_dim.dim_value()) {
        fail_shape_inference("Scale and zero-point grid shapes must match.");
      }
    }
  }
}

static void InferQuantizeLinear29(InferenceContext& ctx) {
  // y_zero_point's type is known only if it is present as an input *and* its type could be inferred
  // (e.g. it is not simply an unresolved formal parameter of an enclosing function).
  const auto* const zp_type = ctx.hasInput(2) ? ctx.getInputType(2) : nullptr;
  auto const output_dtype =
      static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
  if (zp_type != nullptr) {
    auto const zp_elem_type = static_cast<TensorProto_DataType>(getTensorElementType(*zp_type));
    if (output_dtype != TensorProto::UNDEFINED && output_dtype != zp_elem_type) {
      fail_type_inference(
          "output_dtype ",
          TensorProto_DataType_Name(output_dtype),
          " does not match y_zero_point type ",
          TensorProto_DataType_Name(zp_elem_type),
          ".");
    }
    propagateElemTypeFromInputToOutput(ctx, 2, 0);
  } else if (output_dtype != TensorProto::UNDEFINED) {
    propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
  } else if (!ctx.hasInput(2)) {
    // y_zero_point is not provided: the output type defaults to uint8.
    updateOutputElemType(ctx, 0, TensorProto::UINT8);
  }
  // Otherwise, y_zero_point is provided but its type could not be inferred, and no output_dtype
  // attribute was specified: the output type cannot be determined, so it is left uninferred.
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }
  ValidateQuantizationBlockShape(ctx);
}

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    29,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get `y`. For per-tensor/layer quantization the scale is a scalar, for "
            "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the same shape as the "
            "input, except for one dimension in which blocking is performed. With `block_shape`, "
            "it is a grid with dimension `ceil(x.shape[i] / block_shape[i])` along each axis `i`.",
            "T2")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get `y`. Shape must match `y_scale`. "
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T3",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input `x`.", "T3")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis and blocked "
            "quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, "
            "rendering the axis unnecessary in this scenario. Ignored when `block_shape` is specified.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 quantization "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description. "
            "It has no effect for float4e2m1, float6e2m3, or float6e3m2, since those types have no "
            "non-saturating (infinity-representable) encoding to fall back to.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "block_size",
            "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for "
            "blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, "
            "`y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is "
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`. Must be zero when `block_shape` is specified.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "block_shape",
            "(Optional) Multi-axis block sizes, with one positive integer per input dimension. "
            "When specified, y_scale and y_zero_point must have the same rank as x, with dimension "
            "ceil(x.shape[i] / block_shape[i]) along each axis i. Partial edge blocks are supported. "
            "Use 1 for an unblocked dimension. axis is ignored and block_size must be zero.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "output_dtype",
            "(Optional) The output data type. If not supplied, the output data type is inferred from `y_zero_point` data type (`T3`). "
            "If neither `output_dtype` nor `y_zero_point` are supplied, output data type is uint8. "
            "If both `output_dtype` and `y_zero_point` are specified, `output_dtype` must be `T3`.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "precision",
            "(Optional) The precision of the division operation between `x` and `y_scale`. If not provided, "
            "it will be the same as the type of `y_scale`.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T1",
            {types::Float, types::Float16, types::BFloat16, types::Int32},
            "The type of the input 'x'.")
        .TypeConstraint(
            "T2",
            {types::Float, types::Float16, types::BFloat16, types::Int32, types::Float8E8M0},
            "The type of the input 'y_scale'.")
        .TypeConstraint(
            "T3",
            {types::Int8,
             types::UInt8,
             types::Int16,
             types::UInt16,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ,
             types::UInt4,
             types::Int4,
             types::Float4E2M1,
             types::UInt2,
             types::Int2,
             types::Float6E2M3,
             types::Float6E3M2},
            "The type of the input `y_zero_point` and the output `y`.")
        .SetDoc(std::string(kDoc_QuantizeLinear_ver28) + R"DOC(
When `block_shape` is specified, multi-axis blocked quantization is used instead of the
scalar, per-axis, or single-axis blocked modes described above. For input shape
`[D0, ..., Dn]` and `block_shape=[B0, ..., Bn]`, the scale and optional zero point
must have shape `[ceil(D0/B0), ..., ceil(Dn/Bn)]`. At input index `[i0, ..., in]`,
the scale and zero point are selected at `[floor(i0/B0), ..., floor(in/Bn)]`.
Edge blocks may be smaller than `block_shape`. `axis` is ignored and `block_size`
must be zero. Omitting `block_shape` preserves the existing quantization modes.
Rounding, saturation, division precision, output types, and zero-point behavior
are unchanged.

For example, an input of shape `[256, 256]` with `block_shape=[128, 128]`
and FP32 scales `[[1, 2], [4, 8]]` uses one scale per 128 by 128 tile.
An all-ones input quantized to FP8 produces quadrants containing 1, 0.5, 0.25, and 0.125.
For stacked experts `[E, O, I]`, use `block_shape=[1, 128, 128]` and scales
`[E, ceil(O/128), ceil(I/128)]` to keep the experts independent.
)DOC")
        .TypeAndShapeInferenceFunction(InferQuantizeLinear29));

static void InferDequantizeLinear29(InferenceContext& ctx) {
  const auto output_dtype =
      static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
  if (output_dtype != TensorProto::UNDEFINED) {
    propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
  } else {
    propagateElemTypeFromInputToOutput(ctx, 1, 0);
  }
  if (hasInputShape(ctx, 0)) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
  }

  ValidateQuantizationBlockShape(ctx);
}

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    29,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
        .Input(
            1,
            "x_scale",
            "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
            "per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has the same shape as "
            "the input, except for one dimension in which blocking is performed. With `block_shape`, "
            "it is a grid with dimension `ceil(x.shape[i] / block_shape[i])` along each axis `i`.",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input `x`. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T1",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "N-D full precision output tensor. It has the same shape as input `x`. The data type is specified "
            "by the `output_dtype` attribute or, in its absence, the type of `x_scale`.",
            "T3")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used for per-axis and blocked "
            "quantization. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`. Ignored when `block_shape` is specified.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "block_size",
            "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for "
            "blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, "
            "`y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is "
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`. Must be zero when `block_shape` is specified.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "block_shape",
            "(Optional) Multi-axis block sizes, with one positive integer per input dimension. "
            "When specified, x_scale and x_zero_point must have the same rank as x, with dimension "
            "ceil(x.shape[i] / block_shape[i]) along each axis i. Partial edge blocks are supported. "
            "Use 1 for an unblocked dimension. axis is ignored and block_size must be zero.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "output_dtype",
            "(Optional) The output data type. If not supplied, the output data type is inferred from `x_scale` data type (`T2`)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T1",
            {types::Int8,
             types::UInt8,
             types::Int16,
             types::UInt16,
             types::Int32,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ,
             types::UInt4,
             types::Int4,
             types::Float4E2M1,
             types::UInt2,
             types::Int2,
             types::Float6E2M3,
             types::Float6E3M2},
            "The type of the inputs 'x_zero_point' and 'x'.")
        .TypeConstraint(
            "T2",
            {types::Float, types::Float16, types::BFloat16, types::Float8E8M0},
            "The type of the input 'x_scale'.")
        .TypeConstraint("T3", {types::Float, types::Float16, types::BFloat16}, "The type of the output 'y'.")
        .SetDoc(std::string(kDoc_DequantizeLinear_ver24) + R"DOC(
When `block_shape` is specified, multi-axis blocked dequantization is used instead of the
scalar, per-axis, or single-axis blocked modes described above. For input shape
`[D0, ..., Dn]` and `block_shape=[B0, ..., Bn]`, the scale and optional zero point
must have shape `[ceil(D0/B0), ..., ceil(Dn/Bn)]`. At input index `[i0, ..., in]`,
the scale and zero point are selected at `[floor(i0/B0), ..., floor(in/Bn)]`.
Edge blocks may be smaller than `block_shape`. `axis` is ignored and `block_size`
must be zero. Omitting `block_shape` preserves the existing quantization modes.
The data types, zero-point behavior, and arithmetic precision are unchanged.

For example, an FP8 input of shape `[256, 256]` with `block_shape=[128, 128]`
and FP32 scales `[[1, 2], [3, 4]]` uses one scale per 128 by 128 tile.
An all-ones input produces four quadrants containing 1, 2, 3, and 4.
For stacked experts `[E, O, I]`, use `block_shape=[1, 128, 128]` and scales
`[E, ceil(O/128), ceil(I/128)]` to keep the experts independent.
)DOC")
        .TypeAndShapeInferenceFunction(InferDequantizeLinear29));

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    11,
    OpSchema()
        .SetDoc(kDoc_DynamicQuantizeLinear_ver11)
        .Input(0, "x", "Input tensor", "T1")
        .Output(0, "y", "Quantized output tensor", "T2")
        .Output(
            1,
            "y_scale",
            "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Output(
            2,
            "y_zero_point",
            "Output zero point. It's a scalar, which means a per-tensor/layer quantization.",
            "T2")
        .TypeConstraint("T1", {types::Float}, "Constrain 'x' to float tensor.")
        .TypeConstraint("T2", {types::UInt8}, "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
        .FunctionBody(R"ONNX(
        {
           Q_Min = Constant<value = float {0.0}>()
           Q_Max = Constant<value = float {255.0}>()
           One = Constant<value = float {1.0}>()
           X_Min = ReduceMin <keepdims = 0> (x)
           X_Min_Adjusted = Min (X_Min, Q_Min)
           X_Max = ReduceMax <keepdims = 0> (x)
           X_Max_Adjusted = Max (X_Max, Q_Min)
           X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
           X_Range_Is_Empty = Equal (X_Range, Q_Min)
           Unguarded_Scale = Div (X_Range, Q_Max)
           Scale = Where (X_Range_Is_Empty, One, Unguarded_Scale)
           Min_Scaled = Div (X_Min_Adjusted, Scale)
           Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
           Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
           Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
           Zeropoint = Cast <to = 2> (Rounded_ZeroPoint_FP)
           y_scale = Identity (Scale)
           y_zero_point = Identity (Zeropoint)
           y = QuantizeLinear (x, Scale, Zeropoint)
        }
        )ONNX")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::UINT8);
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);
          updateOutputElemType(ctx, 2, TensorProto::UINT8);

          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
