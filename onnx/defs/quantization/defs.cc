// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

namespace {

std::pair<int64_t, int64_t> GGUFBlockSize(int64_t type) {
  switch (type) {
    case 2:
      return {32, 18};
    case 3:
      return {32, 20};
    case 6:
      return {32, 22};
    case 7:
      return {32, 24};
    case 8:
      return {32, 34};
    case 9:
      return {32, 36};
    case 10:
      return {256, 84};
    case 11:
      return {256, 110};
    case 12:
      return {256, 144};
    case 13:
      return {256, 176};
    case 14:
      return {256, 210};
    case 15:
      return {256, 292};
    case 16:
      return {256, 66};
    case 17:
      return {256, 74};
    case 18:
      return {256, 98};
    case 19:
      return {256, 50};
    case 20:
      return {32, 18};
    case 21:
      return {256, 110};
    case 22:
      return {256, 82};
    case 23:
      return {256, 136};
    case 29:
      return {256, 56};
    case 34:
      return {256, 54};
    case 35:
      return {256, 66};
    case 39:
      return {32, 17};
    case 40:
      return {64, 36};
    case 41:
      return {128, 18};
    case 42:
      return {64, 18};
    default:
      fail_shape_inference("Unsupported quantized ggml_type: ", type);
  }
}

void DequantizeGGUFInference(InferenceContext& ctx) {
  const auto* type = ctx.getAttribute("ggml_type");
  const auto* shape = ctx.getAttribute("shape");
  if (type == nullptr || shape == nullptr) {
    fail_shape_inference("DequantizeGGUF requires ggml_type and shape.");
  }
  const auto block = GGUFBlockSize(type->i());
  const auto output_dtype = getAttribute(ctx, "output_dtype", TensorProto::FLOAT);
  if (output_dtype != TensorProto::FLOAT && output_dtype != TensorProto::FLOAT16 &&
      output_dtype != TensorProto::BFLOAT16 && output_dtype != TensorProto::DOUBLE) {
    fail_type_inference("output_dtype must be FLOAT, FLOAT16, BFLOAT16, or DOUBLE.");
  }
  updateOutputElemType(ctx, 0, static_cast<int32_t>(output_dtype));
  if (shape->ints_size() == 0) {
    fail_shape_inference("shape must have rank at least one.");
  }
  bool empty = false;
  for (const auto dim : shape->ints()) {
    if (dim < 0) {
      fail_shape_inference("shape dimensions must be nonnegative; -1 is not supported.");
    }
    empty = empty || dim == 0;
  }
  if (shape->ints(shape->ints_size() - 1) % block.first != 0) {
    fail_shape_inference("The last output dimension must be a multiple of the quantization block size.");
  }
  int64_t bytes = empty ? 0 : block.second;
  for (int i = 0; i < shape->ints_size(); ++i) {
    const auto dim = i == shape->ints_size() - 1 ? shape->ints(i) / block.first : shape->ints(i);
    if (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim) {
      fail_shape_inference("The packed byte count exceeds int64 range.");
    }
    bytes *= dim;
  }
  if (hasInputShape(ctx, 0)) {
    const auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() != 1) {
      fail_shape_inference("The packed input must be a one-dimensional UINT8 tensor.");
    }
    if (input_shape.dim(0).has_dim_value() && input_shape.dim(0).dim_value() != bytes) {
      fail_shape_inference("Packed input length must be exactly ", bytes, " bytes.");
    }
  }
  auto* output_shape = getOutputShape(ctx, 0);
  for (const auto dim : shape->ints()) {
    output_shape->add_dim()->set_dim_value(dim);
  }
}

} // namespace

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeGGUF,
    29,
    OpSchema()
        .SetDoc(R"DOC(
Dequantizes a GGUF tensor's packed quantized payload to a floating-point tensor.
`x` is a one-dimensional UINT8 tensor of consecutive quantization blocks.
All multibyte fields are little-endian.

`ggml_type` selects the tensor's quantization format using the numeric GGML
tensor type identifier. The supported identifiers and block sizes are:

| Identifier | Format | Elements | Bytes |
|---|---|---|---|
| 2 | Q4_0 | 32 | 18 |
| 3 | Q4_1 | 32 | 20 |
| 6 | Q5_0 | 32 | 22 |
| 7 | Q5_1 | 32 | 24 |
| 8 | Q8_0 | 32 | 34 |
| 9 | Q8_1 | 32 | 36 |
| 10 | Q2_K | 256 | 84 |
| 11 | Q3_K | 256 | 110 |
| 12 | Q4_K | 256 | 144 |
| 13 | Q5_K | 256 | 176 |
| 14 | Q6_K | 256 | 210 |
| 15 | Q8_K | 256 | 292 |
| 16 | IQ2_XXS | 256 | 66 |
| 17 | IQ2_XS | 256 | 74 |
| 18 | IQ3_XXS | 256 | 98 |
| 19 | IQ1_S | 256 | 50 |
| 20 | IQ4_NL | 32 | 18 |
| 21 | IQ3_S | 256 | 110 |
| 22 | IQ2_S | 256 | 82 |
| 23 | IQ4_XS | 256 | 136 |
| 29 | IQ1_M | 256 | 56 |
| 34 | TQ1_0 | 256 | 54 |
| 35 | TQ2_0 | 256 | 66 |
| 39 | MXFP4 | 32 | 17 |
| 40 | NVFP4 | 64 | 36 |
| 41 | Q1_0 | 128 | 18 |
| 42 | Q2_0 | 64 | 18 |

The formats use the packed field layouts and IQ codebooks from
[GGML revision da8e2a54b919557b816b1c2beb432ccb766f7ce0](https://github.com/ggml-org/ggml/tree/da8e2a54b919557b816b1c2beb432ccb766f7ce0).
These are defined in
[`src/ggml-common.h`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-common.h);
decoding follows its `dequantize_row_*` functions in
[`src/ggml-quants.c`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-quants.c)
and floating-point scale conversions in
[`src/ggml-impl.h`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-impl.h).
Q8_1 and Q8_K decode as `d * qs[j]` using their scale `d` and signed INT8
values `qs`. Q8_1 uses a FLOAT16 scale; Q8_K uses a FLOAT scale.
Their cached sum fields are ignored.

`shape` gives the logical output dimensions in ONNX (row-major) order.
For a GGUF descriptor `[ne0, ne1, ...]`, use `shape=[..., ne1, ne0]`.
Decoding preserves the flattened element order.
Rank must be at least one. Dimensions are nonnegative literal sizes;
a zero dimension produces an empty tensor. The last dimension must be a
multiple of the format's elements per block, even for an empty tensor.
If B is elements per block and S is bytes per block, the input length must be
`product(shape[:-1]) * (shape[-1] / B) * S`.

Blocks are decoded in input order to FLOAT using the specified GGML
arithmetic and codebooks. The concatenated values are reshaped to `shape`
and converted once to `output_dtype`. Decoding arithmetic uses FLOAT
precision for all output types. FLOAT16 and BFLOAT16 conversion uses
round-to-nearest, ties-to-even, as in Cast.

For example, a Q4_0 block is a little-endian FLOAT16 scale `d` followed
by 16 bytes `q`. It produces `y[j]=d*((q[j]&15)-8)` and
`y[j+16]=d*((q[j]>>4)-8)` for j=0,...,15. Thus 36 bytes containing two
blocks can be decoded with `ggml_type=2, shape=[2,32]`.
)DOC")
        .Input(0, "x", "Consecutive quantization blocks with little-endian fields.", "tensor(uint8)")
        .Output(0, "y", "Dequantized tensor with the specified shape and output_dtype.", "T")
        .Attr("ggml_type", "Quantized GGML tensor type identifier from the supported table.", AttributeProto::INT, true)
        .Attr("shape", "Logical output dimensions in ONNX order; rank at least one.", AttributeProto::INTS, true)
        .Attr(
            "output_dtype",
            "ONNX TensorProto data type of the output: FLOAT, FLOAT16, BFLOAT16, or DOUBLE.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::FLOAT))
        .TypeConstraint("T", {types::Float, types::Float16, types::BFloat16, types::Double}, "Floating-point output.")
        .TypeAndShapeInferenceFunction(DequantizeGGUFInference));

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    28,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get `y`. For per-tensor/layer quantization the scale is a scalar, for "
            "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the same shape as the "
            "input, except for one dimension in which blocking is performed.",
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
            "rendering the axis unnecessary in this scenario.",
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
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`",
            AttributeProto::INT,
            static_cast<int64_t>(0))
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
        .SetDoc(kDoc_QuantizeLinear_ver28)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // y_zero_point's type is known only if it is present as an input *and* its type could be inferred
          // (e.g. it is not simply an unresolved formal parameter of an enclosing function).
          auto const zp_type = ctx.hasInput(2) ? ctx.getInputType(2) : nullptr;
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
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    28,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
        .Input(
            1,
            "x_scale",
            "Scale for input `x`. For per-tensor/layer dequantization the scale is a scalar, for "
            "per per-axis dequantization it is a 1-D Tensor and for blocked dequantization it has the same shape as "
            "the input, except for one dimension in which blocking is performed.",
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
            "where `r = rank(input)`.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "block_size",
            "(Optional) The size of the quantization block (number of times every scale is replicated). Used only for "
            "blocked quantization. The block size is a positive integer. Given `x` shape `(D0, ..., Di, ..., Dn)`, "
            "`y_scale` shape `(S0, ... Si, ...Sn)` and `axis=i`, the accepted range is "
            "`[ceil(Di/Si), ceil(Di/(Si-1))-1]`",
            AttributeProto::INT,
            static_cast<int64_t>(0))
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
        .SetDoc(kDoc_DequantizeLinear_ver24)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto const output_dtype =
              static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
          if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
          } else {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

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
