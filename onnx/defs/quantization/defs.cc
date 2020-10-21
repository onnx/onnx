// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver13_doc = R"DOC(
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor. The scale factor can be a scalar
(per-tensor/layer quantization), or a 1-D tensor for per-axis quantization. The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
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
            "Zero point for doing quantization to get 'y'. It can be a scalar, which means a per-tensor/layer quantization, "
            "or a 1-D tensor for per-axis quantization. "
            "Default value is uint8 typed 0 if it's not specified.",
            "T2",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "N-D quantized output tensor. It has same shape as input 'x'.",
            "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(int32)"},
            "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
              propagateElemTypeFromInputToOutput(ctx, 2, 0);

              if (!hasInputShape(ctx, 0))
                return;

              auto& input_shape = getInputShape(ctx, 0);
              updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver13_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantizations.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
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
            "Zero point for input 'x'. It can be a scalar, which means a per-tensor/layer dequantization, "
            "or a 1-D tensor for per-axis dequantization. "
            "It's optional. 0 is the default value when it's not specified.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "N-D full precision output tensor. It has same shape as input 'x'.",
            "tensor(float)")
        .Attr(
            "axis",
            "(Optional) The axis of the dequantizing dimension of the input tensor. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver13_doc)
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
              auto y_type = ctx.getOutputType(0);
              // only float is supported
              y_type->mutable_tensor_type()->set_elem_type(
                  ONNX_NAMESPACE::TensorProto::FLOAT);

              if (!hasInputShape(ctx, 0))
                return;

              auto& input_shape = getInputShape(ctx, 0);
              updateOutputShape(ctx, 0, input_shape);
            }));

static const char* DynamicQuantizeLinear_ver13_doc = R"DOC(
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
    13,
    OpSchema()
        .SetDoc(DynamicQuantizeLinear_ver13_doc)
        .Input(0, "x", "Input tensor", "T1")
        .Output(0, "y", "Quantized output tensor", "T2")
        .Output(1, "y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.", "tensor(float)")
        .Output(2, "y_zero_point", "Output zero point. It's a scalar, which means a per-tensor/layer quantization.", "T2")
        .Attr(
            "axis",
            "(Optional) The axis of the quantization dimension of the input tensor. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint(
          "T1",
          {"tensor(float)"},
          "Constrain 'x' to float tensor.")
        .TypeConstraint(
          "T2",
          {"tensor(uint8)"},
          "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx,
                const OpSchema& schema,
                FunctionProto& functionProto) -> bool {
                const auto axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : static_cast<int64_t>(1);
                static std::vector<int64_t> single_axes = std::vector<int64_t>({axis});
                auto func_nodes = FunctionBodyHelper::BuildNodes(
                {// nodes: {outputs, op, inputs, attributes}
                FunctionBodyHelper::Const<float>("Q_Min", 0.f),
                FunctionBodyHelper::Const<float>("Q_Max", 255.f),
                {{"X_Min"}, "ReduceMin", {"x"}, {MakeAttribute("keepdims", int64_t(0)), MakeAttribute("axes", single_axes), MakeAttribute("complement_axes", int64_t(1))}},
                {{"X_Min_Adjusted"}, "Min", {"X_Min", "Q_Min"}},
                {{"X_Max"}, "ReduceMax", {"x"}, {MakeAttribute("keepdims", int64_t(0)), MakeAttribute("axes", single_axes), MakeAttribute("complement_axes", int64_t(1))}},
                {{"X_Max_Adjusted"}, "Max", {"X_Max", "Q_Min"}},
                {{"X_Range"}, "Sub", {"X_Max_Adjusted", "X_Min_Adjusted"}},
                {{"Scale"}, "Div", {"X_Range", "Q_Max"}},
                {{"Min_Scaled"}, "Div", {"X_Min_Adjusted", "Scale"}},
                {{"Initial_ZeroPoint_FP"}, "Sub", {"Q_Min", "Min_Scaled"}},
                {{"Clipped_ZeroPoint_FP"}, "Clip", {"Initial_ZeroPoint_FP", "Q_Min", "Q_Max"}},
                {{"Rounded_ZeroPoint_FP"}, "Round", {"Clipped_ZeroPoint_FP"}},
                {{"Zeropoint"}, "Cast", {"Rounded_ZeroPoint_FP"}, {MakeAttribute("to", int64_t(2))}},
                {{"y_scale"}, "Identity", {"Scale"}},
                {{"y_zero_point"}, "Identity", {"Zeropoint"}},
                {{"y"}, "QuantizeLinear", {"x", "Scale", "Zeropoint"}, {MakeRefAttribute("axis", AttributeProto::INT)}}});

                for (const auto& node : func_nodes) {
                    auto new_node = functionProto.add_node();
                    new_node->CopyFrom(node);
                }

                schema.BuildFunction(functionProto);
                return true;
            }));
} // namespace ONNX_NAMESPACE
