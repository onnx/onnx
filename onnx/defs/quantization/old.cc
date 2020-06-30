// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* QuantizeLinear_ver10_doc = R"DOC(
The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
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
        .Output(
            0,
            "y",
            "N-D quantized output tensor. It has same shape as input 'x'.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(int32)"},
            "Constrain 'x' to float or int32 tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.")
        .SetDoc(QuantizeLinear_ver10_doc)
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
              propagateElemTypeFromInputToOutput(ctx, 2, 0);

              if (!hasInputShape(ctx, 0))
                return;

              auto& input_shape = getInputShape(ctx, 0);
              updateOutputShape(ctx, 0, input_shape);
        }));

static const char* DequantizeLinear_ver10_doc = R"DOC(
The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape.
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
        .Output(
            0,
            "y",
            "N-D full precision output tensor. It has same shape as input 'x'.",
            "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(int8)", "tensor(uint8)", "tensor(int32)"},
            "Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.")
        .SetDoc(DequantizeLinear_ver10_doc)
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

} // namespace ONNX_NAMESPACE
