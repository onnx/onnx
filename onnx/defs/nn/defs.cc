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
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.";
static std::string auto_pad_doc =
    "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
    "only intended to support legacy uses, and for framework authors, one is explicitly "
    "encouraged to use explicit padding specified in the pads attribute.";
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> PoolOpSchemaGenerator(
    const char* name,
    const char* opName,
    const char* additionalDescription) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i] + 1)
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{opName}", opName);
    ReplaceAll(doc, "{additionalDescription}", additionalDescription);
    schema.SetDoc(doc);
    schema.Attr(
        "kernel_shape",
        "The size of the kernel along each axis.",
        AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each axis. If not present, the stride defaults to 1 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc.c_str(),
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc.c_str(), AttributeProto::INTS, OPTIONAL);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimension are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size.",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(AveragePool)
    .FillUsing(PoolOpSchemaGenerator(
        "AveragePool",
        "average",
        "The output of each pooling window is divided by the number of elements exclude pad."));

ONNX_OPERATOR_SCHEMA(MaxPool).FillUsing(PoolOpSchemaGenerator(
    "MaxPool",
    "max",
    "The output of each pooling window is maximum number of elements exclude pad."));

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> LpPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.SinceVersion(2);
    schema.Attr(
        "kernel_shape",
        "The size of the kernel along each axis.",
        AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each axis. If not present, the stride defaults to 0 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc.c_str(),
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc.c_str(), AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "p",
        "p value of the Lp norm used to pool over the input data, default is 2.",
        AttributeProto::INT,
        static_cast<int64_t>(2));
    schema.Input(
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
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(LpPool).FillUsing(LpPoolOpSchemaGenerator("LpPool"));

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 ROI {name} pool consumes an input tensor X and region of interests (RoIs) to
 apply {name} pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.Attr(
        "pooled_shape",
        "ROI pool output shape (height, width).",
        AttributeProto::INTS);
    schema.Attr(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling, default is 1.0f.",
        AttributeProto::FLOAT,
        1.f);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data.",
        "T");
    schema.Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should "
        "be a 2-D tensor of shape (num_rois, 5) given as "
        "[[batch_id, x1, y1, x2, y2], ...].",
        "T");
    schema.Output(
        0,
        "Y",
        "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
  };
}

ONNX_OPERATOR_SCHEMA(MaxRoiPool).FillUsing(RoiPoolOpSchemaGenerator("max"));
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> ConvOpSchemaGenerator(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
    ReplaceAll(doc, "{filter_desc}", filter_desc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; "
        "has size (N x C x H x W), where N is the batch size, "
        "C is the number of channels, and H and W are the "
        "height and width. Note that this is for the 2D image."
        "Otherwise the size is (N x D1 x D2 ... x Dn)",
        "T");
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (M x C x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (M x C x k1 x k2 x ... x kn), "
        "where is the dimension of the kernel",
        "T");
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the "
        "convolution. The output dimensions are functions "
        "of the kernel size, stride size, and pad lengths.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "dilations",
        "dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides",
        "Stride along each axis. If not present, the stride defaults to 1 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc.c_str(),
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc.c_str(), AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into, default is 1.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
  };
}

ONNX_OPERATOR_SCHEMA(Conv).FillUsing(ConvOpSchemaGenerator("a filter"));

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(
    const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.)DOC";
    ReplaceAll(doc, "{filter_desc}", filter_desc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image."
        "Otherwise the size is (N x D1 x D2 ... x Dn)",
        "T");
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (C x M x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (C x M x k1 x k2 x ... x kn), "
        "where is the dimension of the kernel",
        "T");
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of C.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "and pad lengths.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "output_shape",
        "The shape of the output."
        " output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +"
        " kernel_shape[i] - pads[start_i] - pads[end_i]",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "output_padding",
        "The zero-padding added to one side of the output."
        " This is also called adjs/adjustment in some frameworks."
        " If output_shape is set, this attribute will be ignored.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "dilations",
        "dilation value along each axis of the filter. If not present, the dilation defaults to 1 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides",
        "Stride along each axis. If not present, the stride defaults to 1 along each axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc.c_str(),
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc.c_str(), AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into, default is 1.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
  };
}

ONNX_OPERATOR_SCHEMA(ConvTranspose)
    .FillUsing(ConvTransposeOpSchemaGenerator("a filter"));

} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator(
    const char* op_type,
    const char* op) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across the
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
    ReplaceAll(doc, "{op_type}", op_type);
    ReplaceAll(doc, "{op}", op);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimension are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. Dimensions will be N x C x 1 x 1",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.SetDoc(doc);
  };
}
ONNX_OPERATOR_SCHEMA(GlobalAveragePool)
    .FillUsing(GlobalPoolingOpSchemaGenerator("AveragePool", "average"));
ONNX_OPERATOR_SCHEMA(GlobalMaxPool)
    .FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max"));
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator(
    const char* op_type,
    const char* op) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across the
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
    ReplaceAll(doc, "{op_type}", op_type);
    ReplaceAll(doc, "{op}", op);
    schema.SetDoc(doc);
    schema.SinceVersion(2);
    schema.Attr(
        "p",
        "p value of the Lp norm used to pool over the input data, default is 2.",
        AttributeProto::INT,
        static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimension are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. Dimensions will be N x C x 1 x 1",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.SetDoc(doc);
  };
}

ONNX_OPERATOR_SCHEMA(GlobalLpPool)
    .FillUsing(GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool"));
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SCHEMA(BatchNormalization)
    .SinceVersion(6)
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
    .SinceVersion(6)
    .SetDoc(R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC")
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

ONNX_OPERATOR_SCHEMA(LpNormalization)
    .Input(0, "input", "Input matrix", "T")
    .Output(0, "output", "Matrix after normalization", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .SetDoc(R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC")
    .Attr(
        "axis",
        "(int64, default -1) the axis on which to apply normalization, -1 mean last axis.",
        AttributeProto::INT,
        static_cast<int64_t>(-1))
    .Attr(
        "p",
        "(int64, default 2) the order of the normalization, only 1 or 2 are supported.",
        AttributeProto::INT,
        static_cast<int64_t>(2));

ONNX_OPERATOR_SCHEMA(Dropout)
    .SinceVersion(6)
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

ONNX_OPERATOR_SCHEMA(Flatten)
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC")
    .Input(0, "input", "A tensor of rank >= axis.", "T")
    .Output(
        0,
        "output",
        "A 2D tensor with the contents of the input tensor, "
        "with input dimensions up to axis flattened to the outer dimension "
        "of the output and remaining input dimensions flattened into the inner "
        "dimension of the output.",
        "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .Attr(
        "axis",
        "(Default to 1) Indicate up to which input dimensions "
        "(exclusive) should be flattened to the outer dimension of the output. "
        "The value for axis must be in the range [0, R], where R is the rank of the input tensor. "
        "When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), "
        "where the shape of the input tensor is (d_0, d_1, ... d_n). ",
        AttributeProto::INT,
        static_cast<int64_t>(1));

ONNX_OPERATOR_SCHEMA(LRN)
    .Attr("size", "The number of channels to sum over", AttributeProto::INT)
    .Attr("alpha", "Scaling parameter", AttributeProto::FLOAT)
    .Attr("beta", "The exponent", AttributeProto::FLOAT)
    .Attr("bias", "Default to 1.f", AttributeProto::FLOAT, 1.0f)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output "
        " types to float tensors.")
    .SetDoc(R"DOC(
Local Response Normalization. It normalizes over local input regions.
Each input value is divided by
(bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta.
)DOC");
