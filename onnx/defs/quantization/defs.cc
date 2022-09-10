/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver13_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
)DOC";

bool BuildContextDependentFunctionBodyQuantizeLinear(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);

  auto* input_type = ctx.getInputType(0);
  if (!input_type->has_tensor_type()) {
    fail_schema("Invalid x input type. Expected a tensor type");
  }
  auto input_elt_type = input_type->tensor_type().elem_type();

  auto output_elt_type = TensorProto::UINT8;
  float quantized_min = 0, quantized_max = 255;

  if (ctx.hasInput(2)) {
    auto* zero_point_type = ctx.getInputType(2);
    if (!zero_point_type->has_tensor_type()) {
      fail_schema("Invalid y_zero_point input type. Expected a tensor type");
    }
    if (zero_point_type->tensor_type().elem_type() == TensorProto::UINT8) {
      quantized_min = 0;
      quantized_max = 255;
      output_elt_type = TensorProto::UINT8;
    } else if (zero_point_type->tensor_type().elem_type() == TensorProto::INT8) {
      quantized_min = -128;
      quantized_max = 127;
      output_elt_type = TensorProto::INT8;
    } else {
      fail_schema("Invalid zero_point_type. Expected TensorProto::UINT8 or TensorProto::INT8");
    }
  }

  bool is_per_axis = ctx.getInputType(1)->tensor_type().shape().dim().size() == 1
    && ctx.getInputType(1)->tensor_type().shape().dim(0).dim_value() > 1;

  builder.Const("quantized_min", ToTensor(quantized_min))
    .Const("quantized_max", ToTensor(quantized_max));
  if (is_per_axis) {
    auto* axis_attribute = ctx.getAttribute("axis");
    int axis = axis_attribute ? ctx.getAttribute("axis")->i() : 1;
    int x_rank = ctx.getInputType(0)->tensor_type().shape().dim().size();
    std::vector<int64_t> y_scale_shape(x_rank - axis, 1);
    y_scale_shape[0] = ctx.getInputType(1)->tensor_type().shape().dim(0).dim_value();
    builder.Const("y_scale_shape", y_scale_shape);
    builder.Add("y_scale_reshape = Reshape (y_scale, y_scale_shape)");
    builder.Add("x_s = Div (x, y_scale_reshape)");
    builder.Add("y_zero_point_reshape = Reshape (y_zero_point, y_scale_shape)");
    builder.Add("y_zero_point_cast = Cast(y_zero_point_reshape)", "to", (int64_t)(input_elt_type));
  } else {
    builder.Add("x_s = Div (x, y_scale)");
    builder.Add("y_zero_point_cast = Cast(y_zero_point)", "to", (int64_t)(input_elt_type));
  }

  builder.Add("x_s_z = Add (x_s, y_zero_point_cast)")
    .Add("x_s_z_clipped = Clip (x_s_z, quantized_min, quantized_max)")
    .Add("y = Cast (x_s_z_clipped)", "to", (int64_t)(output_elt_type));

  schema.BuildFunction(functionProto);
  return true;
}

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
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyQuantizeLinear)
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
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
)DOC";

bool BuildContextDependentFunctionBodyDequantizeLinear(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);

  auto* input_type = ctx.getInputType(0);
  if (!input_type->has_tensor_type()) {
    fail_schema("Invalid x input type. Expected a tensor type");
  }
  auto input_elt_type = input_type->tensor_type().elem_type();
  auto output_elt_type = TensorProto::FLOAT;

  bool is_per_axis = ctx.getInputType(1)->tensor_type().shape().dim().size() == 1
    && ctx.getInputType(1)->tensor_type().shape().dim(0).dim_value() > 1;

  // need to cast to output element type first for cases
  // where quantized types are not accepted as input with some ops.
  builder.Add("x_cast = Cast(x)", "to", (int64_t)(output_elt_type));
  builder.Add("x_zero_point_cast = Cast(x_zero_point)", "to", (int64_t)(output_elt_type));
  if (is_per_axis) {
    auto* axis_attribute = ctx.getAttribute("axis");
    int axis = axis_attribute ? ctx.getAttribute("axis")->i() : 1;
    int x_rank = ctx.getInputType(0)->tensor_type().shape().dim().size();
    std::vector<int64_t> x_scale_shape(x_rank - axis, 1);
    x_scale_shape[0] = ctx.getInputType(1)->tensor_type().shape().dim(0).dim_value();
    builder.Const("x_scale_shape", x_scale_shape);
    builder.Add("x_scale_reshape = Reshape (x_scale, x_scale_shape)");
    builder.Add("x_zero_point_reshape = Reshape (x_zero_point_cast, x_scale_shape)");
    builder.Add("x_sub_zero_point = Sub (x_cast, x_zero_point_reshape)");
    builder.Add("y = Mul (x_sub_zero_point, x_scale_reshape)");
  } else {
    builder.Add("x_sub_zero_point = Sub (x_cast, x_zero_point_cast)");
    builder.Add("y = Mul (x_sub_zero_point, x_scale)");
  }

  schema.BuildFunction(functionProto);
  return true;
}

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
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyDequantizeLinear)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DynamicQuantizeLinear_ver11_doc = R"DOC(
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
 y_scale = (max(x) - min(x))/(qmax - qmin)
 * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
 * data range is adjusted to include 0.
```
Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    11,
    OpSchema()
        .SetDoc(DynamicQuantizeLinear_ver11_doc)
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
        .TypeConstraint("T1", {"tensor(float)"}, "Constrain 'x' to float tensor.")
        .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
        .FunctionBody(R"ONNX(
        {
           Q_Min = Constant<value = float {0.0}>()
           Q_Max = Constant<value = float {255.0}>()
           X_Min = ReduceMin <keepdims = 0> (x)
           X_Min_Adjusted = Min (X_Min, Q_Min)
           X_Max = ReduceMax <keepdims = 0> (x)
           X_Max_Adjusted = Max (X_Max, Q_Min)
           X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
           Scale = Div (X_Range, Q_Max)
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
