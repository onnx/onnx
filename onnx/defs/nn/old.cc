// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
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

ONNX_OPERATOR_SCHEMA(LpPool)
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

ONNX_OPERATOR_SCHEMA(GlobalLpPool)
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
