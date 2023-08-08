/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

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
            "(Optional) The axis of the dequantizing dimension of the input tensor. Ignored for per-tensor quantization. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).",
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
            "'y_scale' determines the output type.")
        .SetDoc(DequantizeLinear_ver19_doc)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto y_type = ctx.getOutputType(0);
          // only float is supported
          y_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DynamicQuantizeLinear_ver20_doc = R"DOC(
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (max(x) - min(x))/(qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8, or [-f8_max, f8_max]
  for any float 8 types
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

y_zero_point must be 0 for any float 8 type.
)DOC";

bool BuildContextDependentFunctionBodyDynamicQuantizeLinear(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  auto to_attr = ctx.getAttribute("to");
  int64_t to = to_attr == nullptr ? 2 : to_attr->i();

  auto mktensori = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
    // tp.add_dims(1);
    return tp;
  };

  auto mktensorf = [](float val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<float>{val});
    // tp.add_dims(1);
    return tp;
  };

  FunctionBuilder builder(functionProto);
  if (to == TensorProto_DataType_UINT8 || to == TensorProto_DataType_INT8) {
    float qmin = to == TensorProto_DataType_UINT8 ? 0.0 : -127.0;
    float qmax = to == TensorProto_DataType_UINT8 ? 255.0 : 127.0;

    builder.Add("zerof = Constant()", "value", mktensorf(0));
    builder.Add("Q_Min32 = Constant()", "value", mktensorf(qmin));
    builder.Add("Q_Max32 = Constant()", "value", mktensorf(qmax));
    builder.Add("Zero = CastLike (zerof, x)");
    builder.Add("Q_Min = CastLike (Q_Min32, x)");
    builder.Add("Q_Max = CastLike (Q_Max32, x)");
    builder.Add("X_Min = ReduceMin <keepdims = 0> (x)");
    builder.Add("X_Min_Adjusted = Min (X_Min, Zero)");
    builder.Add("X_Max = ReduceMax <keepdims = 0> (x)");
    builder.Add("X_Max_Adjusted = Max (X_Max, Zero)");
    builder.Add("X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)");
    builder.Add("Q_Range = Sub (Q_Max, Q_Min)");
    builder.Add("Scale = Div (X_Range, Q_Range)");
    builder.Add("Min_Scaled = Div (X_Min_Adjusted, Scale)");
    builder.Add("Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)");
    builder.Add("Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)");
    builder.Add("Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)");
    builder.Add("Zeropoint = Cast (Rounded_ZeroPoint_FP)", "to", to);
    builder.Add("y_scale = Identity (Scale)");
    builder.Add("y_zero_point = Identity (Zeropoint)");
    builder.Add("y = QuantizeLinear (x, Scale, Zeropoint)");
  } else {
    // float 8 types
    // standard deviation of all finite float 8 values
    float std8 = 1.0f;
    switch (to) {
      case TensorProto_DataType_FLOAT8E4M3FN:
        std8 = 100.057724f;
        break;
      case TensorProto_DataType_FLOAT8E4M3FNUZ:
        std8 = 54.26635f;
        break;
      case TensorProto_DataType_FLOAT8E5M2:
        std8 = 9535.286f;
        break;
      case TensorProto_DataType_FLOAT8E5M2FNUZ:
        std8 = 9403.499f;
        break;
    }
    builder.Add("zeroi = Constant()", "value", mktensori(0));
    builder.Add("Zeropoint = Cast(zeroi)", "to", to);
    builder.Add("xsquare = Mul( x, x )");
    builder.Add("Dev = ReduceMean <keepdims = 0> ( xsquare )");
    builder.Add("Scale = Sqrt(Dev)");
    builder.Add("stdf = Constant()", "value", mktensorf(std8));
    builder.Add("std = CastLike(stdf, Scale)");
    builder.Add("ScaleScaled = Div( Scale, std )");
    builder.Add("y_scale = Identity (ScaleScaled)");
    builder.Add("y_zero_point = Identity (Zeropoint)");
    builder.Add("y = QuantizeLinear (x, ScaleScaled, Zeropoint)");
  }
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    DynamicQuantizeLinear,
    20,
    OpSchema()
        .SetDoc(DynamicQuantizeLinear_ver20_doc)
        .Attr(
            "to",
            "(Optional) The data type to which the elements of the input tensor are quantized. "
            "Default is UINT8.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "x", "Input tensor", "T1")
        .Output(0, "y", "Quantized output tensor", "T2")
        .Output(1, "y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.", "T1")
        .Output(
            2,
            "y_zero_point",
            "Output zero point. It's a scalar, which means a per-tensor/layer quantization.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "Constrain 'x' to float tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(uint8)",
             "tensor(int8)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer or float 8 tensor.")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyDynamicQuantizeLinear)
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          auto to_attr = ctx.getAttribute("to");
          int64_t to = to_attr == nullptr ? 2 : to_attr->i();
          updateOutputElemType(ctx, 0, to);
          updateOutputElemType(ctx, 1, ctx.getInputType(0)->tensor_type().elem_type());
          updateOutputElemType(ctx, 2, to);

          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
          ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();

          if (!hasInputShape(ctx, 0))
            return;

          auto& input_shape = getInputShape(ctx, 0);
          updateOutputShape(ctx, 0, input_shape);
        }));

} // namespace ONNX_NAMESPACE
