// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;

namespace onnx {
    std::function<void(OpSchema&)> AveragePoolOpSchemaGenerator(const char* name) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 {name} consumes an input tensor X and applies average pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Average pooling consisting of averaging all values of a subset of the
 input tensor according to the kernel size and downsampling the
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
            schema.Attr("pads",
                        "The number of zeros padding added to each axis.",
                        AttrType::INTS);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; dimensions for image case "
                         "are (N x C x H x W), where N is the batch size, C is the number of channels, "
                         "and H and W are the height and the width of the data. For non image case, the "
                         "dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.");
            schema.Output(0,
                          "Y",
                          "Output data tensor from average pooling across the input "
                          "tensor. Dimensions will vary based on various kernel, stride, and pad "
                          "sizes.");
        };
    }

    OPERATOR_SCHEMA(AveragePool)
        .FillUsing(AveragePoolOpSchemaGenerator("AveragePool"));

} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> MaxPoolOpSchemaGenerator(const char* name) {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(
 {name} consumes an input tensor X and applies max pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Average pooling consisting of averaging all values of a subset of the
 input tensor according to the kernel size and downsampling the
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
            schema.Attr("pads",
                        "The number of zeros padding added to each axis.",
                        AttrType::INTS);
            schema.Attr("dilations",
                        "Dilation along each axis, 1 means no dilation.",
                        AttrType::INTS);
            schema.Input(0,
                         "X",
                         "Input data tensor from the previous operator; dimensions for image case "
                         "are (N x C x H x W), where N is the batch size, C is the number of channels, "
                         "and H and W are the height and the width of the data. For non image case, the "
                         "dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.");
            schema.Output(0,
                          "Y",
                          "Output data tensor from max pooling across the input "
                          "tensor. Dimensions will vary based on various kernel, stride, and pad "
                          "sizes.");
        };
    }

    OPERATOR_SCHEMA(MaxPool)
        .FillUsing(MaxPoolOpSchemaGenerator("MaxPool"));

} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> ConvOpSchemaGenerator() {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(Compute N-Dimensional convolution on the input.)DOC";
            schema.SetDoc(doc);
            schema.NumInputs(2, 3);
            schema.NumOutputs(1);
            schema.Input(0,
                         "X",
                         "Input data tensor from previous layer; has size (N x C x H x W)"
                         ", where N is the batch size, C is the number of channels, and"
                         " H and W are the height and width. Note that this is for the 2D image."
                         "Otherwise the size is (N x D1 x D2 ... x Dn)");
            schema.Input(1,
                         "weights",
                         "The weight tensor that will be used in the convolutions; "
                         "has size (M x C x kH x kW), where C is the number of channels, "
                         "and kH and kW are the height and width of the kernel, and M is the number "
                         "of feature maps. For more than 2 dimensions, the kernel shape will be "
                         "(M x C x k1 x k2 x ... x kn), where is the dimension of the kernel");
            schema.Output(0,
                          "Y",
                          "Output data tensor that contains the result of the convolution. The "
                          "output dimensions are functions of the kernel size, stride size, "
                          "and pad lengths.");
            schema.Attr("kernel_shape",
                        "The shape of the convolution kernel.",
                         AttrType::INTS);
            schema.Attr("dilations",
                        "dilation value along each axis of the filter.",
                        AttrType::INTS);
            schema.Attr("strides",
                        "stride along each axis.",
                        AttrType::INTS);
            schema.Attr("pads",
                        "The number of zeros padding added to each axis.",
                        AttrType::INTS);
            schema.Attr("group",
                        "number of groups input channels and output channels are divided into",
                        AttrType::INT);
        };
    }

    OPERATOR_SCHEMA(Conv)
        .FillUsing(ConvOpSchemaGenerator());

} // namespace onnx

namespace onnx {
    std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator() {
        return [=](OpSchema& schema) {
            std::string doc = R"DOC(Compute N-Dimensional transposed convolution on the input.)DOC";
            schema.SetDoc(doc);
            schema.NumInputs(2);
            schema.NumOutputs(1);
            schema.Input(0,
                         "X",
                         "Input data tensor from previous layer; has size (N x C x H x W)"
                         ", where N is the batch size, C is the number of channels, and"
                         " H and W are the height and width. Note that this is for the 2D image."
                         "Otherwise the size is (N x D1 x D2 ... x Dn)");
            schema.Input(1,
                         "weights",
                         "The weight tensor that will be used in the convolutions; "
                         "has size (C x M x kH x kW), where C is the number of channels, "
                         "and kH and kW are the height and width of the kernel, and M is the number "
                         "of feature maps. For more than 2 dimensions, the kernel shape will be "
                         "(C x M x k1 x k2 x ... x kn), where is the dimension of the kernel");
            schema.Output(0,
                          "Y",
                          "Output data tensor that contains the result of the convolution. The "
                          "output dimensions are functions of the kernel size, stride size, "
                          "and pad lengths.");
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
            schema.Attr("pads",
                        "The number of zeros padding added to each axis.",
                        AttrType::INTS);
        };
    }

    OPERATOR_SCHEMA(ConvTranspose)
        .FillUsing(ConvTransposeOpSchemaGenerator());

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
                         "Input data tensor from the previous operator; dimensions for image case "
                         "are (N x C x H x W), where N is the batch size, C is the number of channels, "
                         "and H and W are the height and the width of the data. For non image case, the "
                         "dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.");
            schema.Output(0,
                          "Y",
                          "Output data tensor from pooling across the input "
                          "tensor. Dimensions will be N x C x 1 x 1");
            schema.SetDoc(doc);
        };
    }
  OPERATOR_SCHEMA(GlobalAveragePool)
  .FillUsing(GlobalPoolingOpSchemaGenerator("AveragePool", "average"));
  OPERATOR_SCHEMA(GlobalMaxPool)
  .FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max"));
} // namespace onnx

OPERATOR_SCHEMA(BatchNormalization)
    .NumInputs(5)
    .NumOutputs({1, 5})
    .EnforceConsumed({{3, 1}, {4, 2}})
    .SetDoc(R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
)DOC")
    .Attr("spatial",
          "Compute the mean and variance across all spatial elements or per feature.",
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
           "on the order parameter.")
    .Input(1,
           "scale",
           "The scale as a 1-dimensional tensor of size C to be applied to the "
           "output.")
    .Input(2,
           "bias",
           "The bias as a 1-dimensional tensor of size C to be applied to the "
           "output.")
    .Input(3,
           "mean",
           "The running mean (training) or the estimated mean (testing) "
           "as a 1-dimensional tensor of size C.")
    .Input(4,
           "var",
           "The running variance (training) or the estimated "
           "variance (testing) as a 1-dimensional tensor of size C.")
    .Output(0, "Y", "The output 4-dimensional tensor of the same shape as X.")
    .Output(1,
            "mean",
            "The running mean after the BatchNormalization operator. Must be in-place "
            "with the input mean. Should not be used for testing.")
    .Output(2,
            "var",
            "The running variance after the BatchNormalization operator. Must be "
            "in-place with the input var. Should not be used for testing.")
    .Output(3,
            "saved_mean",
            "Saved mean used during training to speed up gradient "
            "computation. Should not be used for testing.")
    .Output(4,
            "saved_var",
            "Saved variance used during training to speed up "
            "gradient computation. Should not be used for testing.");

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
    .Input(0, "data", "The input data as Tensor.")
    .Output(0, "output", "The output.")
    .Output(1, "mask",
               "The output mask. If is_test is nonzero, this output is not filled.");

OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC")
    .Input(0, "input", "A tensor of rank >= axis.")
    .Output(
        0,
        "output",
        "A 2D tensor with the contents of the input tensor, "
        "with input dimensions up to axis flattened to the outer dimension "
        "of the output and remaining input dimensions flattened into the inner "
        "dimension of the output.")
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
    .Input(0, "X", "Input tensor")
    .Output(0, "Y", "Output tensor")
    .SetDoc(R"DOC(
Local Response Normalization. It normalizes over local input regions. Each input value is divided by
 (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta.
)DOC");
