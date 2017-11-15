// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace onnx;

using AttrType = onnx::OpSchema::AttrType;

namespace onnx {
    std::function<void(OpSchema&)> PoolOpSchemaGenerator(const char* name, const char* opName) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a 
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";
            ReplaceAll(doc, "{name}", name);
            ReplaceAll(doc, "{opName}", opName);            
            schema.SetDoc(doc);
            schema.NumInputs(1);
            schema.NumOutputs(1);
            schema.Attr("kernel_shape",
                        "The size of the kernel along each axis.",
                        AttrType::INTS);
            schema.Attr("strides",
                        "Stride along each axis.",
                        AttrType::INTS);
            schema.Attr("auto_pad",
                        "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
                        "SAME_UPPER or SAME_LOWER mean pad the input so that the ouput size match the input."
                        "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
                        "begining for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
                        "only intended to support legacy uses, and for framework authors, one is explicitly "
                        "encouraged to use explicit padding specified in the pads attribute.",
                        AttrType::STRING);
            schema.Attr("pads",
                        "Padding for lower and upper side along each axis, it can take any value greater "
                        "than or equal to 0. The value represent the number of pixels added to the lower "
                        "and upper part of the corresponding axis. `pads` format should be as follow "
                        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
                        "added at the lower part of axis `i` and xi_end, the number of pixels added at "
                        "the upper part of axis `i`. This attribute cannot be used simultaneously with "
                        "auto_pad attribute.",
                        AttrType::INTS);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; "
                         "dimensions for image case are (N x C x H x W), "
                         "where N is the batch size, C is the number of "
                         "channels, and H and W are the height and the "
                         "width of the data. For non image case, the "
                         "dimension are in the form of "
                         "(N x C x D1 x D2 ... Dn), where N is the batch "
                         "size.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor from average or max pooling across "
                          "the input tensor. Dimensions will vary based "
                          "on various kernel, stride, and pad sizes.", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
        };
    }

    OPERATOR_SCHEMA(AveragePool)
        .FillUsing(PoolOpSchemaGenerator("AveragePool", "average"));

    OPERATOR_SCHEMA(MaxPool)
        .FillUsing(PoolOpSchemaGenerator("MaxPool", "max"));
        
} // namespace onnx

namespace onnx {
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
            schema.NumInputs(1);
            schema.NumOutputs(1);
            schema.Attr("kernel_shape",
                        "The size of the kernel along each axis.",
                        AttrType::INTS);
            schema.Attr("strides",
                        "Stride along each axis.",
                        AttrType::INTS);
            schema.Attr("auto_pad",
                        "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
                        "SAME_UPPER or SAME_LOWER mean pad the input so that the ouput size match the input."
                        "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
                        "begining for SAME_LOWER. VALID mean no padding, therefore, read the pixel values "
                        "from the pads attribute. DEPRECATION NOTE: auto_pad is "
                        "only intended to support legacy uses, and for framework authors, one is explicitly "
                        "encouraged to use explicit padding specified in the pads attribute.",
                        AttrType::STRING);
            schema.Attr("pads",
                        "Padding for lower and upper side along each axis, it can take any value greater "
                        "than or equal to 0. The value represent the number of pixels added to the lower "
                        "and upper part of the corresponding axis. `pads` format should be as follow "
                        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
                        "added at the lower part of axis `i` and xi_end, the number of pixels added at "
                        "the upper part of axis `i`. This attribute cannot be used simultaneously with "
                        "auto_pad attribute.",
                        AttrType::INTS);
            schema.Attr("p",
                        "p value of the Lp norm used to pool over the input data, default is 2.0.",
                        AttrType::FLOAT);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; "
                         "dimensions for image case are (N x C x H x W), "
                         "where N is the batch size, C is the number of "
                         "channels, and H and W are the height and the "
                         "width of the data. For non image case, the "
                         "dimension are in the form of "
                         "(N x C x D1 x D2 ... Dn), where N is the "
                         "batch size.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor from Lp pooling across the input "
                          "tensor. Dimensions will vary based on various kernel, stride, and pad "
                          "sizes.", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
        };
    }

    OPERATOR_SCHEMA(LpPool)
        .FillUsing(LpPoolOpSchemaGenerator("LpPool"));

} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator(const char* name) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 ROI {name} pool consumes an input tensor X and region of interests (RoIs) to 
 apply {name} pooling across each RoI, to produce output 4-D tensor of shape 
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC";
            ReplaceAll(doc, "{name}", name);
            schema.SetDoc(doc);
            schema.NumInputs(2);
            schema.NumOutputs(1);
            schema.Attr("pooled_shape",
                        "ROI pool output shape (height, width).",
                        AttrType::INTS);
            schema.Attr("spatial_scale",
                        "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.",
                        AttrType::FLOAT);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; "
                         "dimensions for image case are (N x C x H x W), "
                         "where N is the batch size, C is the number of "
                         "channels, and H and W are the height and the "
                         "width of the data.", "T");
            schema.Input(1,
                         "rois",
                         "RoIs (Regions of Interest) to pool over. Should "
                         "be a 2-D tensor of shape (num_rois, 5) given as "
                         "[[batch_id, x1, y1, x2, y2], ...].", "T");
            schema.Output(0,
                          "Y",
                          "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
        };
    }

    OPERATOR_SCHEMA(MaxRoiPool)
        .FillUsing(RoiPoolOpSchemaGenerator("max"));
} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> ConvOpSchemaGenerator(const char* filter_desc) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
            ReplaceAll(doc, "{filter_desc}", filter_desc);
            schema.SetDoc(doc);
            schema.NumInputs(2, 3);
            schema.NumOutputs(1);
            schema.Input(0,
                         "X",
                         "Input data tensor from previous layer; "
                         "has size (N x C x H x W), where N is the batch size, "
                         "C is the number of channels, and H and W are the "
                         "height and width. Note that this is for the 2D image."
                         "Otherwise the size is (N x D1 x D2 ... x Dn)", "T");
            schema.Input(1,
                         "weights",
                         "The weight tensor that will be used in the "
                         "convolutions; has size (M x C x kH x kW), where C "
                         "is the number of channels, and kH and kW are the "
                         "height and width of the kernel, and M is the number "
                         "of feature maps. For more than 2 dimensions, the "
                         "kernel shape will be (M x C x k1 x k2 x ... x kn), "
                         "where is the dimension of the kernel", "T");
            schema.Input(2,
                         "bias",
                         "Optional 1D bias to be added to the convolution, has size of M.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor that contains the result of the "
                          "convolution. The output dimensions are functions "
                          "of the kernel size, stride size, and pad lengths.", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
            schema.Attr("kernel_shape",
                        "The shape of the convolution kernel.",
                         AttrType::INTS);
            schema.Attr("dilations",
                        "dilation value along each axis of the filter.",
                        AttrType::INTS);
            schema.Attr("strides",
                        "stride along each axis.",
                        AttrType::INTS);
            schema.Attr("auto_pad",
                        "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
                        "SAME_UPPER or SAME_LOWER mean pad the input so that the ouput size match the input."
                        "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
                        "begining for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
                        "only intended to support legacy uses, and for framework authors, one is explicitly "
                        "encouraged to use explicit padding specified in the pads attribute.",
                        AttrType::STRING);
            schema.Attr("pads",
                        "Padding for lower and upper side along each axis, it can take any value greater "
                        "than or equal to 0. The value represent the number of pixels added to the lower "
                        "and upper part of the corresponding axis. `pads` format should be as follow "
                        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
                        "added at the lower part of axis `i` and xi_end, the number of pixels added at "
                        "the upper part of axis `i`. This attribute cannot be used simultaneously with "
                        "auto_pad attribute.",
                        AttrType::INTS);
            schema.Attr("group",
                        "number of groups input channels and output channels are divided into",
                        AttrType::INT);
        };
    }

    OPERATOR_SCHEMA(Conv)
        .FillUsing(ConvOpSchemaGenerator("a filter"));

} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(const char* filter_desc) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.)DOC";
            ReplaceAll(doc, "{filter_desc}", filter_desc);
            schema.SetDoc(doc);
            schema.NumInputs(2, 3);
            schema.NumOutputs(1);
            schema.Input(0,
                         "X",
                         "Input data tensor from previous layer; has size (N x C x H x W)"
                         ", where N is the batch size, C is the number of channels, and"
                         " H and W are the height and width. Note that this is for the 2D image."
                         "Otherwise the size is (N x D1 x D2 ... x Dn)", "T");
            schema.Input(1,
                         "weights",
                         "The weight tensor that will be used in the "
                         "convolutions; has size (C x M x kH x kW), where C "
                         "is the number of channels, and kH and kW are the "
                         "height and width of the kernel, and M is the number "
                         "of feature maps. For more than 2 dimensions, the "
                         "kernel shape will be (C x M x k1 x k2 x ... x kn), "
                         "where is the dimension of the kernel", "T");
            schema.Input(2,
                         "bias",
                         "Optional 1D bias to be added to the convolution, has size of C.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor that contains the result of the convolution. The "
                          "output dimensions are functions of the kernel size, stride size, "
                          "and pad lengths.", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
            schema.Attr("kernel_shape",
                        "The shape of the convolution kernel.",
                         AttrType::INTS);
            schema.Attr("output_shape",
                        "The shape of the output.",
                        AttrType::INTS);
            schema.Attr("dilations",
                        "dilation value along each axis of the filter.",
                        AttrType::INTS);
            schema.Attr("strides",
                        "stride along each axis.",
                        AttrType::INTS);
            schema.Attr("auto_pad",
                        "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
                        "SAME_UPPER or SAME_LOWER mean pad the input so that the ouput size match the input."
                        "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
                        "begining for SAME_LOWER. VALID mean no padding.",
                        AttrType::STRING);
            schema.Attr("pads",
                        "Padding for lower and upper side along each axis, it can take any value greater "
                        "than or equal to 0. The value represent the number of pixels added to the lower "
                        "and upper part of the corresponding axis. `pads` format should be as follow "
                        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
                        "added at the lower part of axis `i` and xi_end, the number of pixels added at "
                        "the upper part of axis `i`. This attribute cannot be used simultaneously with "
                        "auto_pad attribute.",
                        AttrType::INTS);
            schema.Attr("group",
                        "number of groups input channels and output channels are divided into",
                        AttrType::INT);
        };
    }

    OPERATOR_SCHEMA(ConvTranspose)
        .FillUsing(ConvTransposeOpSchemaGenerator("a filter"));

} // namespace onnx

namespace onnx {
  std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator(const char* op_type, const char* op) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across the
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
            ReplaceAll(doc, "{op_type}", op_type);
            ReplaceAll(doc, "{op}", op);
            schema.SetDoc(doc);
            schema.NumInputs(1);
            schema.NumOutputs(1);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; "
                         "dimensions for image case are (N x C x H x W), "
                         "where N is the batch size, C is the number of "
                         "channels, and H and W are the height and the width "
                         "of the data. For non image case, the dimension are "
                         "in the form of (N x C x D1 x D2 ... Dn), "
                         "where N is the batch size.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor from pooling across the input "
                          "tensor. Dimensions will be N x C x 1 x 1", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
            schema.SetDoc(doc);
        };
    }
  OPERATOR_SCHEMA(GlobalAveragePool)
  .FillUsing(GlobalPoolingOpSchemaGenerator("AveragePool", "average"));
  OPERATOR_SCHEMA(GlobalMaxPool)
  .FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max"));
} // namespace onnx

namespace onnx {
  std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator(const char* op_type, const char* op) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across the
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
            ReplaceAll(doc, "{op_type}", op_type);
            ReplaceAll(doc, "{op}", op);
            schema.SetDoc(doc);
            schema.NumInputs(1);
            schema.NumOutputs(1);
            schema.Attr("p",
                        "p value of the Lp norm used to pool over the input data, default is 2.0.",
                        AttrType::FLOAT);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; "
                         "dimensions for image case are (N x C x H x W), "
                         "where N is the batch size, C is the number of "
                         "channels, and H and W are the height and the width "
                         "of the data. For non image case, the dimension are "
                         "in the form of (N x C x D1 x D2 ... Dn), "
                         "where N is the batch size.", "T");
            schema.Output(0,
                          "Y",
                          "Output data tensor from pooling across the input "
                          "tensor. Dimensions will be N x C x 1 x 1", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
            schema.SetDoc(doc);
        };
    }
  OPERATOR_SCHEMA(GlobalLpPool)
  .FillUsing(GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool"));
} // namespace onnx

OPERATOR_SCHEMA(BatchNormalization)
    .NumInputs(5)
    .NumOutputs({ 1, 5 })
    .EnforceConsumed({ {3, 1}, {4, 2} })
    .SetDoc(R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC")
    .Attr("spatial",
        "If true, compute the mean and variance across all spatial elements "
        "If false, compute the mean and variance across per feature.",
        AttrType::INT)
    .Attr("is_test",
        "If set to nonzero, run spatial batch normalization in test mode.",
        AttrType::INT)
    .Attr("epsilon",
        "The epsilon value to use to avoid division by zero.",
        AttrType::FLOAT)
    .Attr("momentum",
        "Factor used in computing the running mean and variance."
        "e.g., running_mean = running_mean * momentum + mean * (1 - momentum)",
        AttrType::FLOAT)
    .Input(0,
        "X",
        "The input 4-dimensional tensor of shape NCHW or NHWC depending "
        "on the order parameter.", "T")
    .Input(1,
        "scale",
        "The scale as a 1-dimensional tensor of size C to be applied to the "
        "output.", "T")
    .Input(2,
        "bias",
        "The bias as a 1-dimensional tensor of size C to be applied to the "
        "output.", "T")
    .Input(3,
        "mean",
        "The running mean (training) or the estimated mean (testing) "
        "as a 1-dimensional tensor of size C.", "T")
    .Input(4,
        "var",
        "The running variance (training) or the estimated "
        "variance (testing) as a 1-dimensional tensor of size C.", "T")
    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.", "T")
    .Output(1,
        "mean",
        "The running mean after the BatchNormalization operator. Must be in-place "
        "with the input mean. Should not be used for testing.", "T")
    .Output(2,
        "var",
        "The running variance after the BatchNormalization operator. Must be "
        "in-place with the input var. Should not be used for testing.", "T")
    .Output(3,
        "saved_mean",
        "Saved mean used during training to speed up gradient "
        "computation. Should not be used for testing.", "T")
    .Output(4,
        "saved_var",
        "Saved variance used during training to speed up "
        "gradient computation. Should not be used for testing.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(InstanceNormalization)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022. 

y = scale * (x - mean) / sqrt(variance + epsilon) + bias, 
where mean and bias are computed per instance per channel. 

)DOC")
    .Attr("epsilon",
        "The epsilon value to use to avoid division by zero.",
        AttrType::FLOAT)
    .Input(0,
        "input",
        "The input 4-dimensional tensor of shape NCHW.", "T")
    .Input(1,
        "scale",
        "The input 1-dimensional scale tensor of size C.", "T")
    .Input(2,
        "bias",
        "The input 1-dimensional bias tensor of size C.", "T")
    .Output(0,
        "output",
        "The output 4-dimensional tensor of the same shape as input.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(LpNormalization)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "input", "Input matrix", "T")
    .Output(0, "output", "Matrix after normalization", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .SetDoc(R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC")
    .Attr("axis", "(int64, default -1) the axis on which to apply normalization, -1 mean last axis.", AttrType::INT)
    .Attr("p", "(int64, default 2) the order of the normalization, only 1 or 2 are supported.", AttrType::INT);

OPERATOR_SCHEMA(Dropout)
    .NumInputs(1)
    .NumOutputs(1,2)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC")
    .Attr("ratio",
          "(float, default 0.5) the ratio of random dropout",
          AttrType::FLOAT)
    .Attr("is_test",
          "(int, default 0) if nonzero, run dropout in test mode where "
          "the output is simply Y = X.",
          AttrType::INT)
    .Input(0, "data", "The input data as Tensor.", "T")
    .Output(0, "output", "The output.", "T")
    .Output(1, "mask",
               "The output mask. If is_test is nonzero, this output is not filled.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
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
        "dimension of the output.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.")
    .Attr(
        "axis",
        "(Default to 1) Indicate up to which input dimensions "
        "(exclusive) should be flattened to the outer dimension of the output",
        AttrType::INT);

OPERATOR_SCHEMA(LRN)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("size", "The number of channels to sum over", AttrType::INT, true)
    .Attr("alpha", "Scaling parameter", AttrType::FLOAT, true)
    .Attr("beta", "The exponent", AttrType::FLOAT, true)
    .Attr("bias", "Default to 1", AttrType::FLOAT)
    .Input(0, "X", "Input tensor", "T")
    .Output(0, "Y", "Output tensor", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output "
        " types to float tensors.")
    .SetDoc(R"DOC(
Local Response Normalization. It normalizes over local input regions.
Each input value is divided by
(bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta.
)DOC");
