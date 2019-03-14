// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
using SupportType = OpSchema::SupportType;
using SupportType = ONNX_NAMESPACE::OpSchema::SupportType;

// Experimental ops in ONNX do not have version maintained.
// Experimental ops are used for verifying op definitions (experimentation)
// before checked into ONNX or ONNX-ML domain as official ops, and partners
// do not need to implement these ops. An experimental op should be either removed
// or promoted after a while. In this file, a default since_version "1" is used for all exp ops.

static const char* ScaledTanh_ver1_doc = R"DOC(
Calculates the scaled hyperbolic tangent of the given input tensor element-wise,
alpha * tanh(beta * x).
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ScaledTanh,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc(ScaledTanh_ver1_doc)
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The scaled hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    GivenTensorFill,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .Input(
            0,
            "shape",
            "The shape of filled tensor",
            "T",
            OpSchema::Optional)
        .Output(0, "X", "The filled tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .Attr("values", "", AttributeProto::FLOATS, OPTIONAL)
        .Attr("shape", "", AttributeProto::INTS, OPTIONAL)
        .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL)
        .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getAttribute("shape") != nullptr) {
            propagateShapeFromAttributeToOutput(ctx, "shape", 0);
            return;
          }
          // The type constraints above do not allow for input_as_shape
          // and may need to be fixed.
          if (getAttribute(ctx, "input_as_shape", 0) != 0) // dynamic shape
            return;
          std::vector<int64_t> extra_shape;
          getRepeatedAttribute(ctx, "extra_shape", extra_shape);
          if (hasInputShape(ctx, 0)) {
            TensorShapeProto shape = ctx.getInputType(0)->tensor_type().shape();
            for (auto extra_dim_val : extra_shape) {
              if (extra_dim_val < 0)
                fail_shape_inference(
                    "Negative values are not allowed in a shape specification");
              shape.add_dim()->set_dim_value(extra_dim_val);
            }
            updateOutputShape(ctx, 0, shape);
          }
        }));

static const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Scale,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .Input(0, "input", "Input data to be scaled", "T")
        .Output(0, "output", "Output data after scaling", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetDoc(Scale_ver1_doc)
        .Attr(
            "scale",
            "The scale to apply.",
            AttributeProto::FLOAT,
            1.0f)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GRUUnit,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc(GRUUnit_ver1_doc)
        .Attr(
            "drop_states",
            "Bool to determine if hidden state is zeroes or passed "
            "along for timesteps past the given sequence_length.",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
        .Input(
            1,
            "gates",
            "Unactivated gate outputs from forget, update, "
            "and output gates, pre-activation.",
            "T")
        .Input(
            2,
            "seq_lengths",
            "Array of sequence lengths.  "
            "len(seq_lengths) should equal batch size N.",
            "T")
        .Input(3, "t", "The timestep for this operation.", "T")
        .Output(
            0,
            "hidden",
            "The new GRU hidden state calculated by this op.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* ATen_ver1_doc = R"DOC(
Experimental allowing ATen operations to be accessed directly from Caffe2
to allow for quick prototyping when ONNX is missing standard versions of
and op)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ATen,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .AllowUncheckedAttributes()
        .SetDoc(ATen_ver1_doc)
        .Input(0, "input", "Arbitrary input", "T", OpSchema::Variadic)
        .Output(0, "output", "Arbitrary output", "T", OpSchema::Variadic)
        .TypeConstraint(
            "T",
            {"tensor(bool)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(double)"},
            "Constrain output types to bool, int32, int64, float16, float, double tensors."));

static const char* ImageScaler_ver1_doc =
    R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ImageScaler,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc(ImageScaler_ver1_doc)
        .Attr(
            "bias",
            "Bias applied to each channel, same size as C.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "scale",
            "The scale to apply.",
            AttributeProto::FLOAT,
            1.0f)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Crop_ver1_doc =
    R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Crop,
    1,
    OpSchema()
        .SetSupportLevel(SupportType::EXPERIMENTAL)
        .SetDoc(Crop_ver1_doc)
        .Attr(
            "border",
            "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "scale",
            "A 1-D values of (height, width).",
            AttributeProto::INTS,
            OPTIONAL)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(
            0,
            "output",
            "Result, has same type as input, with H and W dimensions reduced.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));
} // namespace ONNX_NAMESPACE
