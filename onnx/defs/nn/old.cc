// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace onnx;

namespace onnx {
    static std::string pads_doc = "Padding for the beginning and ending along each axis, it can take any value greater "
                                  "than or equal to 0. The value represent the number of pixels added to the beginning "
                                  "and end part of the corresponding axis. `pads` format should be as follow "
                                  "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
                                  "added at the beginning of axis `i` and xi_end, the number of pixels added at "
                                  "the end of axis `i`. This attribute cannot be used simultaneously with "
                                  "auto_pad attribute.";
    static std::string auto_pad_doc = "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where "
                                      "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
                                      "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
                                      "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
                                      "only intended to support legacy uses, and for framework authors, one is explicitly "
                                      "encouraged to use explicit padding specified in the pads attribute.";
}

OPERATOR_SCHEMA(LpPool)
    .SinceVersion(1)
    .SetDoc(R"DOC(
 LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC")
    .Attr("kernel_shape",
          "The size of the kernel along each axis.",
          AttributeProto::INTS,
          OPTIONAL)
    .Attr("strides",
          "Stride along each axis.",
          AttributeProto::INTS,
          OPTIONAL)
    .Attr("auto_pad",
          auto_pad_doc.c_str(),
          AttributeProto::STRING,
          std::string("NOTSET"))
    .Attr("pads",
          pads_doc.c_str(),
          AttributeProto::INTS,
          OPTIONAL)
    .Attr("p",
          "p value of the Lp norm used to pool over the input data, default is 2.0.",
          AttributeProto::FLOAT,
          2.0f)
    .Input(0,
           "X",
           "Input data tensor from the previous operator; "
           "dimensions for image case are (N x C x H x W), "
           "where N is the batch size, C is the number of "
           "channels, and H and W are the height and the "
           "width of the data. For non image case, the "
           "dimension are in the form of "
           "(N x C x D1 x D2 ... Dn), where N is the "
           "batch size.", "T")
    .Output(0,
            "Y",
            "Output data tensor from Lp pooling across the input "
            "tensor. Dimensions will vary based on various kernel, stride, and pad "
            "sizes.", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(GlobalLpPool)
    .SinceVersion(1)
    .SetDoc(R"DOC(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)DOC")
    .Attr("p",
          "p value of the Lp norm used to pool over the input data, default is 2.0.",
          AttributeProto::FLOAT,
          2.0f)
    .Input(0,
           "X",
           "Input data tensor from the previous operator; "
           "dimensions for image case are (N x C x H x W), "
           "where N is the batch size, C is the number of "
           "channels, and H and W are the height and the width "
           "of the data. For non image case, the dimension are "
           "in the form of (N x C x D1 x D2 ... Dn), "
           "where N is the batch size.", "T")
    .Output(0,
            "Y",
            "Output data tensor from pooling across the input "
            "tensor. Dimensions will be N x C x 1 x 1", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(AveragePool)
	.SinceVersion(1)
    .SetDoc(R"DOC(
 AveragePool consumes an input tensor X and applies AveragePool pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 AveragePool pooling consisting of computing the AveragePool on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC")
	.Attr("kernel_shape",
			"The size of the kernel along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("strides",
			"Stride along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("auto_pad",
			auto_pad_doc.c_str(),
			AttributeProto::STRING,
			std::string("NOTSET"))
	.Attr("pads",
			pads_doc.c_str(),
			AttributeProto::INTS, OPTIONAL)
	.Input(0,
			"X",
			"Input data tensor from the previous operator; "
			"dimensions for image case are (N x C x H x W), "
			"where N is the batch size, C is the number of "
			"channels, and H and W are the height and the "
			"width of the data. For non image case, the "
			"dimension are in the form of "
			"(N x C x D1 x D2 ... Dn), where N is the batch "
			"size.", "T")
	.Output(0,
			"Y",
			"Output data tensor from average or max pooling across "
			"the input tensor. Dimensions will vary based "
			"on various kernel, stride, and pad sizes.", "T")
	.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
			"Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(MaxPool)
    .SetDoc(R"DOC(
 MaxPool consumes an input tensor X and applies MaxPool pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 MaxPool pooling consisting of computing the MaxPool on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC")
	.SinceVersion(1)
	.Attr("kernel_shape",
			"The size of the kernel along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("strides",
			"Stride along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("auto_pad",
			auto_pad_doc.c_str(),
			AttributeProto::STRING,
			std::string("NOTSET"))
	.Attr("pads",
			pads_doc.c_str(),
			AttributeProto::INTS, OPTIONAL)
	.Input(0,
			"X",
			"Input data tensor from the previous operator; "
			"dimensions for image case are (N x C x H x W), "
			"where N is the batch size, C is the number of "
			"channels, and H and W are the height and the "
			"width of the data. For non image case, the "
			"dimension are in the form of "
			"(N x C x D1 x D2 ... Dn), where N is the batch "
			"size.", "T")
	.Output(0,
			"Y",
			"Output data tensor from average or max pooling across "
			"the input tensor. Dimensions will vary based "
			"on various kernel, stride, and pad sizes.", "T")
	.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
			"Constrain input and output types to float tensors.");


OPERATOR_SCHEMA(LpPool)
    .SetDoc(R"DOC(
 LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC")
	.SinceVersion(2)
	.Attr("kernel_shape",
			"The size of the kernel along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("strides",
			"Stride along each axis.",
			AttributeProto::INTS, OPTIONAL)
	.Attr("auto_pad",
			auto_pad_doc.c_str(),
			AttributeProto::STRING,
			std::string("NOTSET"))
	.Attr("pads",
			pads_doc.c_str(),
			AttributeProto::INTS, OPTIONAL)
	.Attr("p",
			"p value of the Lp norm used to pool over the input data",
			AttributeProto::INT, OPTIONAL)
	.Input(0,
			"X",
			"Input data tensor from the previous operator; "
			"dimensions for image case are (N x C x H x W), "
			"where N is the batch size, C is the number of "
			"channels, and H and W are the height and the "
			"width of the data. For non image case, the "
			"dimension are in the form of "
			"(N x C x D1 x D2 ... Dn), where N is the "
			"batch size.", "T")
	.Output(0,
			"Y",
			"Output data tensor from Lp pooling across the input "
			"tensor. Dimensions will vary based on various kernel, stride, and pad "
			"sizes.", "T")
	.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
			"Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(MaxRoiPool)
    .SetDoc(R"DOC(
 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC")
    .SinceVersion(1)
    .Attr("pooled_shape",
          "ROI pool output shape (height, width).",
          AttributeProto::INTS, OPTIONAL)
    .Attr("spatial_scale",
          "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling",
          AttributeProto::FLOAT,  OPTIONAL)
    .Input(0,
           "X",
           "Input data tensor from the previous operator; "
           "dimensions for image case are (N x C x H x W), "
           "where N is the batch size, C is the number of "
           "channels, and H and W are the height and the "
           "width of the data.", "T")
    .Input(1,
           "rois",
           "RoIs (Regions of Interest) to pool over. Should "
           "be a 2-D tensor of shape (num_rois, 5) given as "
           "[[batch_id, x1, y1, x2, y2], ...].", "T")
    .Output(0,
            "Y",
            "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).", "T")
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
        "Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(Conv)
    .SetDoc(R"DOC(
The convolution operator consumes an input tensor and a filter, and
computes the output.)DOC")
    .SinceVersion(1)
    .Input(0,
           "X",
           "Input data tensor from previous layer; "
           "has size (N x C x H x W), where N is the batch size, "
           "C is the number of channels, and H and W are the "
           "height and width. Note that this is for the 2D image."
           "Otherwise the size is (N x D1 x D2 ... x Dn)", "T")
    .Input(1,
           "W",
           "The weight tensor that will be used in the "
           "convolutions; has size (M x C x kH x kW), where C "
           "is the number of channels, and kH and kW are the "
           "height and width of the kernel, and M is the number "
           "of feature maps. For more than 2 dimensions, the "
           "kernel shape will be (M x C x k1 x k2 x ... x kn), "
           "where is the dimension of the kernel", "T")
    .Input(2,
           "B",
           "Optional 1D bias to be added to the convolution, has size of M.", "T", OpSchema::Optional)
    .Output(0,
            "Y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.", "T")
    .Attr("kernel_shape",
          "The shape of the convolution kernel.",
           AttributeProto::INTS, OPTIONAL)
    .Attr("dilations",
          "dilation value along each axis of the filter.",
          AttributeProto::INTS, OPTIONAL)
    .Attr("strides",
          "stride along each axis.",
          AttributeProto::INTS, OPTIONAL)
    .Attr("auto_pad",
          auto_pad_doc.c_str(),
          AttributeProto::STRING,
          std::string("NOTSET"))
    .Attr("pads",
          pads_doc.c_str(),
          AttributeProto::INTS, OPTIONAL)
	.Attr("group",
		"number of groups input channels and output channels are divided into.",
		AttributeProto::INT, OPTIONAL)
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
			"Constrain input and output types to float tensors.");

OPERATOR_SCHEMA(ConvTranspose)
    .SetDoc(R"DOC(
The convolution transpose operator consumes an input tensor and filter,
and computes the output.)DOC")
    .SinceVersion(1)
    .Input(0,
           "X",
           "Input data tensor from previous layer; has size (N x C x H x W)"
           ", where N is the batch size, C is the number of channels, and"
           " H and W are the height and width. Note that this is for the 2D image."
           "Otherwise the size is (N x D1 x D2 ... x Dn)", "T")
    .Input(1,
           "W",
           "The weight tensor that will be used in the "
           "convolutions; has size (C x M x kH x kW), where C "
           "is the number of channels, and kH and kW are the "
           "height and width of the kernel, and M is the number "
           "of feature maps. For more than 2 dimensions, the "
           "kernel shape will be (C x M x k1 x k2 x ... x kn), "
           "where is the dimension of the kernel", "T")
    .Input(2,
           "B",
           "Optional 1D bias to be added to the convolution, has size of C.", "T", OpSchema::Optional)
    .Output(0,
            "Y",
            "Output data tensor that contains the result of the convolution. The "
            "output dimensions are functions of the kernel size, stride size, "
            "and pad lengths.", "T")
    .Attr("kernel_shape",
          "The shape of the convolution kernel.",
           AttributeProto::INTS, OPTIONAL)
    .Attr("output_shape",
          "The shape of the output."
          " output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +"
          " kernel_shape[i] - pads[start_i] - pads[end_i]",
          AttributeProto::INTS, OPTIONAL)
    .Attr("output_padding",
          "The zero-padding added to one side of the output."
          " This is also called adjs/adjustment in some frameworks."
          " If output_shape is set, this attribute will be ignored.",
          AttributeProto::INTS, OPTIONAL)
    .Attr("dilations",
          "dilation value along each axis of the filter.",
          AttributeProto::INTS, OPTIONAL)
    .Attr("strides",
          "stride along each axis.",
          AttributeProto::INTS, OPTIONAL)
    .Attr("auto_pad",
          auto_pad_doc.c_str(),
          AttributeProto::STRING,
          std::string("NOTSET"))
    .Attr("pads",
          pads_doc.c_str(),
          AttributeProto::INTS, OPTIONAL)
    .Attr("group",
          "number of groups input channels and output channels are divided into.",
          AttributeProto::INT, OPTIONAL)
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
          "Constrain input and output types to float tensors.");
