// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
static std::string pads_doc =
    "Padding for the beginning and ending along each axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute.";
static std::string auto_pad_doc =
    "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
    "only intended to support legacy uses, and for framework authors, one is explicitly "
    "encouraged to use explicit padding specified in the pads attribute.";
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SCHEMA(LpPool)
    .SinceVersion(1)
    .SetDoc(R"DOC(
 LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC")
    .Attr(
        "kernel_shape",
        "The size of the kernel along each axis.",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr("strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL)
    .Attr(
        "auto_pad",
        auto_pad_doc.c_str(),
        AttributeProto::STRING,
        std::string("NOTSET"))
    .Attr("pads", pads_doc.c_str(), AttributeProto::INTS, OPTIONAL)
    .Attr(
        "p",
        "p value of the Lp norm used to pool over the input data, default is 2.0.",
        AttributeProto::FLOAT,
        2.0f)
    .Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimension are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T")
    .Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(GlobalLpPool)
    .SinceVersion(1)
    .SetDoc(R"DOC(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)DOC")
    .Attr(
        "p",
        "p value of the Lp norm used to pool over the input data, default is 2.0.",
        AttributeProto::FLOAT,
        2.0f)
    .Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimension are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T")
    .Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. Dimensions will be N x C x 1 x 1",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(BatchNormalization)
    .SinceVersion(1)
    .NumOutputs({1, 5})
    .SetDoc(R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC")
    .Attr(
        "spatial",
        "If true, compute the mean and variance across all spatial elements "
        "If false, compute the mean and variance across per feature."
        "Default is 1.",
        AttributeProto::INT,
        static_cast<int64_t>(1))
    .Attr(
        "is_test",
        "If set to nonzero, run spatial batch normalization in test mode, default is 0.",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Attr(
        "epsilon",
        "The epsilon value to use to avoid division by zero, default is 1e-5f.",
        AttributeProto::FLOAT,
        1e-5f)
    .Attr(
        "momentum",
        "Factor used in computing the running mean and variance."
        "e.g., running_mean = running_mean * momentum + mean * (1 - momentum), default is 0.9f.",
        AttributeProto::FLOAT,
        0.9f)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS)
    .Input(0, "X", "The input 4-dimensional tensor of shape NCHW.", "T")
    .Input(
        1,
        "scale",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.",
        "T")
    .Input(
        2,
        "B",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.",
        "T")
    .Input(
        3,
        "mean",
        "The running mean (training) or the estimated mean (testing) "
        "as a 1-dimensional tensor of size C.",
        "T")
    .Input(
        4,
        "var",
        "The running variance (training) or the estimated "
        "variance (testing) as a 1-dimensional tensor of size C.",
        "T")
    .Output(
        0,
        "Y",
        "The output 4-dimensional tensor of the same shape as X.",
        "T")
    .Output(
        1,
        "mean",
        "The running mean after the BatchNormalization operator. Must be in-place "
        "with the input mean. Should not be used for testing.",
        "T",
        OpSchema::Optional)
    .Output(
        2,
        "var",
        "The running variance after the BatchNormalization operator. Must be "
        "in-place with the input var. Should not be used for testing.",
        "T",
        OpSchema::Optional)
    .Output(
        3,
        "saved_mean",
        "Saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.",
        "T",
        OpSchema::Optional)
    .Output(
        4,
        "saved_var",
        "Saved variance used during training to speed up "
        "gradient computation. Should not be used for testing.",
        "T",
        OpSchema::Optional)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(InstanceNormalization)
    .SetDoc(R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC")
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr(
        "epsilon",
        "The epsilon value to use to avoid division by zero, default is 1e-5f.",
        AttributeProto::FLOAT,
        1e-5f)
    .Input(0, "input", "The input 4-dimensional tensor of shape NCHW.", "T")
    .Input(1, "scale", "The input 1-dimensional scale tensor of size C.", "T")
    .Input(2, "B", "The input 1-dimensional bias tensor of size C.", "T")
    .Output(
        0,
        "output",
        "The output 4-dimensional tensor of the same shape as input.",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_OPERATOR_SCHEMA(Dropout)
    .SetDoc(R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC")
    .Attr(
        "ratio",
        "(float, default 0.5) the ratio of random dropout",
        AttributeProto::FLOAT,
        0.5f)
    // This attribute was added via AllowConsumed API in OpSchema.
    // After removing the API, we're now using the Attr API to simulate the old
    // definition.
    .Attr(
        "consumed_inputs",
        "legacy optimization attribute.",
        AttributeProto::INTS,
        OPTIONAL)
    .Attr(
        "is_test",
        "(int, default 0) if nonzero, run dropout in test mode where "
        "the output is simply Y = X.",
        AttributeProto::INT,
        static_cast<int64_t>(0))
    .Input(0, "data", "The input data as Tensor.", "T")
    .Output(0, "output", "The output.", "T")
    .Output(
        1,
        "mask",
        "The output mask. If is_test is nonzero, this output is not filled.",
        "T",
        OpSchema::Optional)
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");