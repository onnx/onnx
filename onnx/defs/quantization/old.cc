/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver23_doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.

Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]

For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.

`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 and 4bit types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
`x` and `y_scale` are allowed to have different types. The type of `y_scale` determines the precision of the division operation between `x` and
`y_scale`, unless the `precision` attribute is specified.

There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    23,
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
            "Zero point for doing quantization to get `y`. Shape must match `y_scale`."
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
            "All cases are fully described in two tables inserted in the operator description.",
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
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "The type of the input 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "The type of the input 'y_scale'.")
        .TypeConstraint(
            "T3",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)",
             "tensor(float4e2m1)"},
            "The type of the input `y_zero_point` and the output `y`.")
        .SetDoc(QuantizeLinear_ver23_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver23_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.

`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 and 4-bit types quantization, but the dequantization formula remains the same
for consistency. The output type is determined by the attribute `output_dtype`. If `output_dtype` is not supplied then the output type
is the same as `x_scale`. The output type also determines the precision of the multiplication operation.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    23,
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
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(int32)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)",
             "tensor(float4e2m1)"},
            "The type of the inputs 'x_zero_point' and 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "The type of the input 'x_scale'.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "The type of the output 'y'.")
        .SetDoc(DequantizeLinear_ver23_doc)
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

static const char* QuantizeLinear_ver21_doc = R"DOC(
The linear quantization operator consumes a high-precision tensor, a scale, and a zero point to compute the
low-precision/quantized tensor. The scale factor and zero point must have the same shape, determining the quantization
granularity. The quantization formula is `y = saturate((x / y_scale) + y_zero_point)`.
Saturation is done according to:
- uint16: [0, 65535]
- int16: [-32768, 32767]
- uint8: [0, 255]
- int8: [-128, 127]
- uint4: [0, 15]
- int4: [-8, 7]
For `(x / y_scale)`, it rounds to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
`y_zero_point` and `y` must have the same type. `y_zero_point` is usually not used for quantization to float8 types, but the quantization
formula remains the same for consistency, and the type of the attribute `y_zero_point` still determines the quantization type.
There are three supported quantization granularities, determined by the shape of `y_scale`.
In all cases, `y_zero_point` must have the same shape as `y_scale`.
- Per-tensor (per-layer) quantization: `y_scale` is a scalar.
- Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis. For an input shape
 `(D0, ..., Di, ..., Dn)` and `axis=i`, `y_scale` is a 1-D tensor of length `Di`.
- Blocked quantization: The scale's shape is identical to the input's shape, except for one dimension, in which
  blocking is performed. Given `x` shape `(D0, ..., Di, ..., Dn)`, `axis=i`, and block size `B`: `y_scale` shape is
  `(D0, ..., ceil(Di/B), ..., Dn)`.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    21,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get `y`. For per-tensor/layer quantization the scale is a scalar, for "
            "per-axis quantization it is a 1-D Tensor and for blocked quantization it has the same shape as the "
            "input, except for one dimension in which blocking is performed.",
            "T1")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get `y`. Shape must match `y_scale`."
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input `x`.", "T2")
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
            "All cases are fully described in two tables inserted in the operator description.",
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
            "(Optional) The output data type. If not supplied, the output data type is inferred from `y_zero_point` data type (`T2`). "
            "If neither `output_dtype` nor `y_zero_point` are supplied, output data type is uint8. "
            "If both `output_dtype` and `y_zero_point` are specified, `output_dtype` must be `T2`.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "The type of the input 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)"},
            "The type of the input `y_zero_point` and the output `y`.")
        .SetDoc(QuantizeLinear_ver21_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
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
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver21_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the
full-precision tensor. The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point`
must have the same shape, determining the quantization's granularity: a scalar for per-tensor/per-layer quantization,
a 1-D tensor for per-axis quantization, or have a rank identical to the input for blocked quantization.
See QuantizeLinear for details on quantization granularity.
`x_zero_point` and `x` must have the same type. `x` and `y` must have the same shape. In the case of dequantizing
`int32`, there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8 types quantization, but the dequantization formula remains the same
for consistency, and `x_scale` still determines the output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    21,
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
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input `x`.", "T2")
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
        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int16)",
             "tensor(uint16)",
             "tensor(int32)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(uint4)",
             "tensor(int4)"},
            "The type of the inputs 'x_zero_point' and 'x'.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "'x_scale' determines the output type.")
        .SetDoc(DequantizeLinear_ver21_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* QuantizeLinear_ver19_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    19,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, "
            "or a 1-D Tensor for per-axis quantization.",
            "T1")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 quantization "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)", "tensor(int32)"},
            "Constrain 'x' to float, float16, bfloat16 or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer/float tensor.")
        .SetDoc(QuantizeLinear_ver19_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver19_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    19,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, "
            "or a 1-D tensor for per-axis dequantization.",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T1",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Used only for per-axis quantization. "
            "Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` "
            "where `r = rank(input)`. When the rank of the input is 1, per-tensor quantization is applied, "
            "rendering the axis unnecessary in this scenario.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(int8)",
             "tensor(uint8)",
             "tensor(int32)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit integer or float, or /32-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "'x_scale' determines the output type.")
        .SetDoc(DequantizeLinear_ver19_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* QuantizeLinear_ver13_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    13,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It can be a scalar, which means per-tensor/layer quantization, "
            "or a 1-D Tensor for per-axis quantization.",
            "tensor(float)")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. Shape must match y_scale. "
            "Default is uint8 with zero point of 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint("T1", {"tensor(float)", "tensor(int32)"}, "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver13_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    13,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, "
            "or a 1-D tensor for per-axis dequantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. Shape must match x_scale. "
            "It's optional. Zero point is 0 when it's not specified.",
            "T",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "tensor(float)")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* QuantizeLinear_ver10_doc = R"DOC(
The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QuantizeLinear,
    10,
    OpSchema()
        .Input(0, "x", "N-D full precision Input tensor to be quantized.", "T1")
        .Input(
            1,
            "y_scale",
            "Scale for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "y_zero_point",
            "Zero point for doing quantization to get 'y'. It's a scalar, which means a per-tensor/layer quantization. "
            "Default value is uint8 typed 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(0, "y", "N-D quantized output tensor. It has same shape as input 'x'.", "T2")
        .TypeConstraint("T1", {"tensor(float)", "tensor(int32)"}, "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver10_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.hasInput(2)) {
            propagateElemTypeFromInputToOutput(ctx, 2, 0);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::UINT8);
          }
          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver10_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DequantizeLinear,
    10,
    OpSchema()
        .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T")
        .Input(
            1,
            "x_scale",
            "Scale for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point for input 'x'. It's a scalar, which means a per-tensor/layer quantization. "
            "It's optional. 0 is the default value when it's not specified.",
            "T",
            OpSchema::Optional)
        .Output(0, "y", "N-D full precision output tensor. It has same shape as input 'x'.", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver10_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
