/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <limits>

#include "onnx/common/assertions.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
static const char* pads_doc =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
static const char* conv_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";
static const char* conv_transpose_auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that "
    "`output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. "
    "The padding is split between the two sides equally or almost equally (depending "
    "on whether it is even or odd). In case the padding is an odd number, the extra "
    "padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.";

ONNX_API void convPoolShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape,
    int input1Idx,
    int input2Idx) {
  // we need the first input shape for this inference.
  if (!hasInputShape(ctx, input1Idx)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasInputShape(ctx, input2Idx)) {
    return;
  }

  auto input_shape = ctx.getInputType(input1Idx)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Only MaxPool and Conv support dilation. For
  // simplicity of the code, we just treat the rest of them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size");
    }
  } else if (require_kernel_shape) {
    fail_shape_inference("Attribute kernel_shape must be specified");
  } else {
    auto second_input_shape = ctx.getInputType(input2Idx)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          if (!input_shape.dim(2 + i).has_dim_value()) {
            continue;
          }
          residual = input_shape.dim(2 + i).dim_value();
          while (residual >= stride) {
            residual -= stride;
          }
        }
        if (i >= static_cast<int>(effective_kernel_shape.size())) {
          fail_shape_inference("kernel shape should have ", input_dims_size, " values in ", ctx.getDisplayName(), ".");
        }
        int64_t total_pad = residual == 0 ? effective_kernel_shape[i] - stride : effective_kernel_shape[i] - residual;
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  if (require_kernel_shape) {
    // add the first two dimensions from the input.
    *output_shape->add_dim() = input_shape.dim(0);
    *output_shape->add_dim() = input_shape.dim(1);
  } else {
    *output_shape->add_dim() = input_shape.dim(0);
    auto& second_input_shape = getInputShape(ctx, input2Idx);
    if (second_input_shape.dim_size() < 1) {
      fail_shape_inference("Second input tensor has wrong dimension");
    }
    *output_shape->add_dim() = second_input_shape.dim(0);
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }
    // how big is the input, including padding
    int64_t input_size = input_shape.dim(2 + i).dim_value();
    int64_t effective_input_size = input_size + pads[i] + pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    int64_t output_size =
        (effective_input_size - effective_kernel_shape[i] + (ceil_mode ? strides[i] - 1 : 0)) / strides[i] + 1;
    if (ceil_mode == 1 && (output_size - 1) * strides[i] >= (input_size + pads[i])) {
      // we need to match pytorch's behavior of "Sliding windows that would start in the right padded region are
      // ignored." (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#maxpool1d). this code follows the
      // same logic as PyTorch's C++ implementation:
      // https://github.com/pytorch/pytorch/blob/f1cdb39da3850c47d51ec6a5b1ae864c32b3accf/aten/src/ATen/native/Pool.h#L54C21-L54C21
      --output_size;
    }

    newdim->set_dim_value(output_size);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto second_output_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

static std::vector<std::string> GetSupportedDataTypesForPoolingOps(bool supports8bit) {
  if (supports8bit) {
    return OpSchema::all_float_types_plus_Xint8_ir4();
  }
  return OpSchema::all_float_types_ir4();
}

static std::function<void(OpSchema&)> PoolOpSchemaGenerator(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation,
    bool supports8bit = false) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```
 {additionalDescription}
 )DOC";
        ReplaceAll(doc, "{name}", name);
        ReplaceAll(doc, "{opName}", opName);
        ReplaceAll(doc, "{additionalDescription}", additionalDescription);
        ReplaceAll(
            doc,
            "{kernelSpatialShape}",
            use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)" : "kernel_spatial_shape[i]"););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the batch "
        "size. Optionally, if dimension denotation is "
        "in effect, the operation expects the input "
        "data tensor to arrive with the dimension denotation "
        "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from average or max pooling across "
        "the input tensor. Dimensions will vary based "
        "on various kernel, stride, and pad sizes. Floor value of "
        "the dimension is used",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForPoolingOps(supports8bit),
        supports8bit ? "Constrain input and output types to float and 8 bit tensors."
                     : "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([use_dilation](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (ctx.getNumOutputs() > 1) {
        // MaxPool with two outputs case.
        auto output_type = ctx.getOutputType(1);
        if (output_type->value_case() == TypeProto::kTensorType ||
            output_type->value_case() == TypeProto::VALUE_NOT_SET) {
          output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
        }
      }
      convPoolShapeInference(ctx, use_dilation, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    22,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "AveragePool",
            "average",
            "The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).",
            true, /* use_dilation: dilations attribute has been added in opset 19. */
            false /* supports8bit: does not support 8bit. */))
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "count_include_pad",
            "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
            AttributeProto::INT,
            static_cast<int64_t>(0)));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    22,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "MaxPool",
            "max",
            "The output of each pooling window is maximum number of elements exclude pad. ",
            true,
            true))
        .Attr(
            "storage_order",
            "The storage order of the tensor. 0 is row major, and 1 is column major. "
            "This attribute is used only to convert an n-tuple index value into "
            "a single integer value for producing the second output. ",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Output(
            1,
            "Indices",
            "Indices tensor from max pooling across the input tensor. "
            "The dimensions of indices are the same as output tensor. "
            "The values in indices of are the indices of the selected values during pooling. "
            "The indices are computed as flatten 1-D tensor, "
            "and the indices do not consider padding. "
            "So the values in indices are in [0, N x C x D1 x ... x Dn).",
            "I",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("I", {"tensor(int64)"}, "Constrain index tensor to int64"));

static void maxUnpoolShapeInference(InferenceContext& ctx) {
  // we need at least two inputs to have a shape for this inference.
  if (ctx.getNumInputs() != 2 && ctx.getNumInputs() != 3) {
    fail_type_inference("MaxUnpool op must have either two or three inputs.");
  }
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasInputShape(ctx, 0)) {
    return; // If first input does not have shape, we cannot infer much.
  }
  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor X must have at least 2 dimensions.");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size.");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute kernel_shape has incorrect size.");
    }
  } else {
    fail_shape_inference("Attribute kernel_shape must be specified.");
  }

  if (ctx.getNumInputs() == 3) {
    // If the third input, output_size, is specified, then use that instead
    // of inferring shape from inputs.
    if (hasInputShape(ctx, 2)) {
      auto& output_shape = getInputShape(ctx, 2);
      if (output_shape.dim_size() != 1) {
        fail_type_inference("'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim((int)0).has_dim_value() &&
          static_cast<int>(output_shape.dim((int)0).dim_value()) != input_shape.dim_size()) {
        fail_shape_inference("'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value = strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += kernel_shape[i];
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

static const char* MaxUnpool_ver22_doc = R"DOC(
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    22,
    OpSchema()
        .SetDoc(MaxUnpool_ver22_doc)
        .Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(
            0,
            "X",
            "Input data tensor that has to be unpooled. "
            "This tensor is typically the first output of the MaxPool op."
            "Dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non-image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size. Optionally, if dimension denotation is "
            "in effect, the operation expects the input "
            "data tensor to arrive with the dimension denotation "
            "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "I",
            "Input data tensor containing the indices corresponding to "
            "elements in the first input tensor X."
            "This tensor is typically the second output of the MaxPool op."
            "Dimensions must be the same as input tensor X. "
            "The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, "
            "assuming row-major storage. Also, the linear indices should not consider padding. "
            "So the values in indices are in the range [0, N x C x D1 x ... x Dn).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "output_shape",
            "The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, "
            "'pads' values are ignored.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output data tensor that contains the result of the unpooling.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain index tensor to int64")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { maxUnpoolShapeInference(ctx); }));

static std::function<void(OpSchema&)> LpPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("kernel_shape", "The size of the kernel along each axis.", AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Attr(
        "ceil_mode",
        "Whether to use ceil or floor (default) to compute the output shape.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the "
        "width of the data. For non image case, the "
        "dimensions are in the form of "
        "(N x C x D1 x D2 ... Dn), where N is the "
        "batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from Lp pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, true, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(LpPool, 22, OpSchema().FillUsing(LpPoolOpSchemaGenerator("LpPool")));

// For ROI pool operations.
static void roiPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // rois is the second input.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  auto rios_shape = ctx.getInputType(1)->tensor_type().shape();

  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have at least 2 dimensions");
  }
  if (rios_shape.dim_size() != 2) {
    fail_shape_inference("RoIs tensor must have 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> pooled_shape;
  if (getRepeatedAttribute(ctx, "pooled_shape", pooled_shape)) {
    if (pooled_shape.size() != n_input_dims) {
      fail_shape_inference("Attribute pooled_shape has incorrect length");
    }
  } else {
    fail_shape_inference("Attribute pooled_shape must be specified");
  }

  // (num_rois, channels, pooled_shape[0], pooled_shape[1])
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *output_shape->add_dim() = rios_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);
  output_shape->add_dim()->set_dim_value(pooled_shape[0]);
  output_shape->add_dim()->set_dim_value(pooled_shape[1]);
}

static std::function<void(OpSchema&)> RoiPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 ROI {name} pool consumes an input tensor X and region of interests (RoIs) to
 apply {name} pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).)DOC";
        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("pooled_shape", "ROI pool output shape (height, width).", AttributeProto::INTS);
    schema.Attr(
        "spatial_scale",
        "Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.",
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
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "rois",
        "RoIs (Regions of Interest) to pool over. Should "
        "be a 2-D tensor of shape (num_rois, 5) given as "
        "[[batch_id, x1, y1, x2, y2], ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "Y",
        "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { roiPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(MaxRoiPool, 22, OpSchema().FillUsing(RoiPoolOpSchemaGenerator("max")));

static std::function<void(OpSchema&)> ConvOpSchemaGenerator(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution operator consumes an input tensor and {filter_desc}, and
computes the output.)DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; "
        "has size (N x C x H x W), where N is the batch size, "
        "C is the number of channels, and H and W are the "
        "height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
        "Optionally, if dimension denotation is "
        "in effect, the operation expects input data tensor "
        "to arrive with the dimension denotation of [DATA_BATCH, "
        "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (M x C/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... kn) is the dimension of the kernel. "
        "Optionally, if dimension denotation is in effect, "
        "the operation expects the weight tensor to arrive "
        "with the dimension denotation of [FILTER_OUT_CHANNEL, "
        "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
        "Assuming zero based indices for the shape array, "
        "X.shape[1] == (W.shape[1] * group) == C and "
        "W.shape[0] mod G == 0. Or in other words "
        "FILTER_IN_CHANNEL multiplied by the number of groups "
        "should be equal to DATA_CHANNEL and the number of "
        "feature maps M should be a multiple of the number of "
        "groups G.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the "
        "convolution. The output dimensions are functions "
        "of the kernel size, stride size, and pad lengths.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, true, false, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(Conv, 22, OpSchema().FillUsing(ConvOpSchemaGenerator("a filter")));

static const char* QLinearConv_ver10_doc = R"DOC(
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    QLinearConv,
    10,
    OpSchema()
        .SetDoc(QLinearConv_ver10_doc)
        .Input(
            0,
            "x",
            "Input data tensor from previous layer; "
            "has size (N x C x H x W), where N is the batch size, "
            "C is the number of channels, and H and W are the "
            "height and width. Note that this is for the 2D image. "
            "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
            "Optionally, if dimension denotation is "
            "in effect, the operation expects input data tensor "
            "to arrive with the dimension denotation of [DATA_BATCH, "
            "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1")
        .Input(
            1,
            "x_scale",
            "Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            2,
            "x_zero_point",
            "Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization.",
            "T1")
        .Input(
            3,
            "w",
            "The weight tensor that will be used in the "
            "convolutions; has size (M x C/group x kH x kW), where C "
            "is the number of channels, and kH and kW are the "
            "height and width of the kernel, and M is the number "
            "of feature maps. For more than 2 dimensions, the "
            "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
            "where (k1 x k2 x ... kn) is the dimension of the kernel. "
            "Optionally, if dimension denotation is in effect, "
            "the operation expects the weight tensor to arrive "
            "with the dimension denotation of [FILTER_OUT_CHANNEL, "
            "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
            "X.shape[1] == (W.shape[1] * group) == C "
            "(assuming zero based indices for the shape array). "
            "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
            "T2")
        .Input(
            4,
            "w_scale",
            "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "tensor(float)")
        .Input(
            5,
            "w_zero_point",
            "Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "T2")
        .Input(
            6,
            "y_scale",
            "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            7,
            "y_zero_point",
            "Zero point tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "T3")
        .Input(
            8,
            "B",
            "Optional 1D bias to be added to the convolution, has size of M. "
            "Bias must be quantized using scale = x_scale * w_scale and zero_point = 0",
            "T4",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.",
            "T3")
        .TypeConstraint("T1", {"tensor(int8)", "tensor(uint8)"}, "Constrain input type to 8-bit integer tensor.")
        .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)"}, "Constrain filter type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int8)", "tensor(uint8)"}, "Constrain output type to 8-bit integer tensor.")
        .TypeConstraint("T4", {"tensor(int32)"}, "Constrain bias type to 32-bit integer tensor.")
        .Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(3);
          if (nullptr == x_type || nullptr == w_type || x_type->value_case() != TypeProto::kTensorType ||
              w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type in ", ctx.getDisplayName(), ".");
          }

          auto x_zero_point_type = ctx.getInputType(2);
          if (nullptr == x_zero_point_type ||
              x_zero_point_type->tensor_type().elem_type() != x_type->tensor_type().elem_type()) {
            fail_type_inference(
                "input and zero_point pair is expected to have be same type in ", ctx.getDisplayName(), ".");
          }

          auto w_zero_point_type = ctx.getInputType(5);
          if (nullptr == w_zero_point_type ||
              w_zero_point_type->tensor_type().elem_type() != w_type->tensor_type().elem_type()) {
            fail_type_inference(
                "weight and zero_point pair is expected to have same type in ", ctx.getDisplayName(), ".");
          }

          propagateElemTypeFromInputToOutput(ctx, 7, 0);

          convPoolShapeInference(ctx, true, false, 0, 3);
        }));

static const char* ConvInteger_ver10_doc = R"DOC(
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConvInteger,
    10,
    OpSchema()
        .SetDoc(ConvInteger_ver10_doc)
        .Input(
            0,
            "x",
            "Input data tensor from previous layer; "
            "has size (N x C x H x W), where N is the batch size, "
            "C is the number of channels, and H and W are the "
            "height and width. Note that this is for the 2D image. "
            "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
            "Optionally, if dimension denotation is "
            "in effect, the operation expects input data tensor "
            "to arrive with the dimension denotation of [DATA_BATCH, "
            "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T1")
        .Input(
            1,
            "w",
            "The weight tensor that will be used in the "
            "convolutions; has size (M x C/group x kH x kW), where C "
            "is the number of channels, and kH and kW are the "
            "height and width of the kernel, and M is the number "
            "of feature maps. For more than 2 dimensions, the "
            "kernel shape will be (M x C/group x k1 x k2 x ... x kn), "
            "where (k1 x k2 x ... kn) is the dimension of the kernel. "
            "Optionally, if dimension denotation is in effect, "
            "the operation expects the weight tensor to arrive "
            "with the dimension denotation of [FILTER_OUT_CHANNEL, "
            "FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. "
            "X.shape[1] == (W.shape[1] * group) == C "
            "(assuming zero based indices for the shape array). "
            "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
            "T2")
        .Input(
            2,
            "x_zero_point",
            "Zero point tensor for input 'x'. It's optional and default value is 0. It's a scalar, which means a per-tensor/layer quantization.",
            "T1",
            OpSchema::Optional)
        .Input(
            3,
            "w_zero_point",
            "Zero point tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
            "which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number "
            "of elements should be equal to the number of output channels (M)",
            "T2",
            OpSchema::Optional)
        .Output(
            0,
            "y",
            "Output data tensor that contains the result of the "
            "convolution. The output dimensions are functions "
            "of the kernel size, stride size, and pad lengths.",
            "T3")
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input x and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input w and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint("T3", {"tensor(int32)"}, "Constrain output y data type to 32-bit integer tensor.")
        .Attr("auto_pad", conv_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
              x_type->value_case() != TypeProto::kTensorType || w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(TensorProto::INT32);

          convPoolShapeInference(ctx, true, false, 0, 1);
        }));

ONNX_API void convTransposeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  int64_t group = getAttribute(ctx, "group", 1);

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return; // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      return;
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      return;
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> kernel_shape;
  if (getRepeatedAttribute(ctx, "kernel_shape", kernel_shape)) {
    if (kernel_shape.size() != n_input_dims) {
      return;
    }
  } else {
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
    }
  }

  std::vector<int64_t> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] = (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if (nullptr != auto_pad_attr && auto_pad_attr->s() != "NOTSET") {
      fail_shape_inference("The pads attribute cannot be used simultaneously with auto_pad attribute");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
    const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
    if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t total_pad = effective_kernel_shape[i] - strides[i];
        if (total_pad < 0)
          total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr->s() == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr->s() == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
  }

  std::vector<int64_t> output_shape;
  bool output_shape_presented = true;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
  } else {
    output_shape_presented = false;
  }

  std::vector<int64_t> output_padding;
  if (getRepeatedAttribute(ctx, "output_padding", output_padding)) {
    if (output_padding.size() != n_input_dims) { // Added only to one side.
      return;
    }
  } else {
    output_padding.assign(n_input_dims, 0);
  }

  auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) * group; // channels should be the second dim of second input
                                                                 // multiply group.

  int size_of_output = 0;
  if (output_shape_presented) {
    size_of_output = static_cast<int>(output_shape.size());
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
          // TODO: throw exception?
          return; // output shape value cannot be smaller than the input shape
                  // value
        }
      }
      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return;
  } else {
    size_of_output = input_shape.dim_size() - 2;
    for (int i = 0; i < size_of_output; ++i) {
      if (input_shape.dim(i + 2).has_dim_value()) {
        int64_t output_shape_dim = strides[i] * (input_shape.dim(i + 2).dim_value() - 1) + output_padding[i] +
            effective_kernel_shape[i] - pads[i] - pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

static std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC";
        ReplaceAll(doc, "{filter_desc}", filter_desc););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        1,
        "W",
        "The weight tensor that will be used in the "
        "convolutions; has size (C x M/group x kH x kW), where C "
        "is the number of channels, and kH and kW are the "
        "height and width of the kernel, and M is the number "
        "of feature maps. For more than 2 dimensions, the "
        "weight shape will be (C x M/group x k1 x k2 x ... x kn), "
        "where (k1 x k2 x ... x kn) is the dimension of the kernel. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Input(
        2,
        "B",
        "Optional 1D bias to be added to the convolution, has size of M.",
        "T",
        OpSchema::Optional,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "pad lengths and group count. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.Attr(
        "kernel_shape",
        "The shape of the convolution kernel. If not present, should be inferred from input W.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_shape",
        "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
        "pads values are ignored. See doc for details for equations to generate pads. Note that the output_shape attribute value "
        "should not include dimensions for batch size and channels, which are automatically inferred.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_padding",
        "Additional elements added to the side with higher coordinate indices in the output. "
        "Each padding value in \"output_padding\" must be less than the corresponding stride/dilation dimension. "
        "By default, this attribute is a zero vector. "
        "Note that this attribute doesn't directly affect the computed output values. "
        "It only controls the selection of the computed values, "
        "so changing this attribute only adds or removes output elements. "
        "If \"output_shape\" is explicitly provided, "
        "\"output_padding\" does not contribute additional size to \"output_shape\" but "
        "participates in the computation of the needed padding amount. "
        "This is also called adjs or adjustment in some frameworks.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL_VALUE);
    schema.Attr("auto_pad", conv_transpose_auto_pad_doc, AttributeProto::STRING, std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL_VALUE);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { convTransposeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(ConvTranspose, 22, OpSchema().FillUsing(ConvTransposeOpSchemaGenerator("a filter")));

static const char* DeformConv_ver22_doc = R"DOC(
Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    DeformConv,
    22,
    OpSchema()
        .SetDoc(DeformConv_ver22_doc)
        .Input(
            0,
            "X",
            "Input data tensor. For 2D image data, it has shape (N, C, H, W) where N is the batch size, "
            "C is the number of input channels, and H and W are the height and width. "
            "In general, the shape is (N, C, D1, D2, ... , Dn) for n-dimensional data, where "
            "D1 to Dn are the spatial dimension sizes. Most common use cases have n = 2 or 3.",
            "T")
        .Input(
            1,
            "W",
            "Weight tensor that will be used in the convolutions. It has shape (oC, C/group, kH, kW), "
            "where oC is the number of output channels and kH and kW are the kernel height and width. "
            "For more than 2 dimensions, it has shape (oC, C/group, k1, k2, ... , kn).",
            "T")
        .Input(
            2,
            "offset",
            "Offset tensor denoting the offset for the sampling locations in the convolution kernel. "
            "It has shape (N, offset_group * kH * kW * 2, oH, oW) for 2D data or "
            "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Use linear interpolation"
            "for fractional offset values. Sampling locations outside of the padded input tensor gives zero.",
            "T")
        .Input(
            3,
            "B",
            "Optional 1D bias of length oC to be added to the convolution. Default is a tensor of zeros.",
            "T",
            OpSchema::Optional)
        .Input(
            4,
            "mask",
            "The mask tensor to be applied to each position in the convolution kernel. "
            "It has shape (N, offset_group * kH * kW, oH, oW) for 2D data or "
            "(N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Default is a "
            "tensor of ones.",
            "T",
            OpSchema::Optional)
        .Output(
            0,
            "Y",
            "Output data tensor that contains the result of convolution. It has shape (N, oC, oH, oW) "
            "for 2D data or (N, oC, o1, o2, ..., on) for nD data",
            "T")
        .TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.")
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of the kernel. Default is 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "group",
            "Number of groups the input and output channels, C and oC, are divided into. C and oC must both "
            "be divisible by group. Default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "kernel_shape",
            "Shape of the convolution kernel. If not present, it is inferred from the shape of input W.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "offset_group",
            "Number of groups of offset. C must be divisible by offset_group. Default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "pads",
            "Padding for the beginning and end along each spatial axis. The values represent the number of pixels "
            "added to the beginning and end of the corresponding axis and can take any nonnegative value. "
            "The format should be as follows: [x1_begin, x2_begin, ..., x1_end, x2_end, ...], where xi_begin "
            "is the number of pixels added at the beginning of axis `i` and xi_end is the number of pixels "
            "added at the end of axis `i`. Default is 0 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "Stride along each spatial axis. Default is 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          convPoolShapeInference(ctx, true, false, 0, 1);
        }));

// For GlobalPool operations.
ONNX_API void globalPoolTypeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // needs at least one input with shape.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return;
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // (N, C, 1, 1, ..., 1)
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

static std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
        ReplaceAll(doc, "{op_type}", op_type);
        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    22,
    OpSchema().FillUsing(GlobalPoolingOpSchemaGenerator("AveragePool", "average")));
ONNX_OPERATOR_SET_SCHEMA(GlobalMaxPool, 22, OpSchema().FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max")));

static std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator(const char* op_type, const char* op) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(
        doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
        ReplaceAll(doc, "{op_type}", op_type);
        ReplaceAll(doc, "{op}", op););
    schema.SetDoc(doc);
    schema.Attr(
        "p", "p value of the Lp norm used to pool over the input data.", AttributeProto::INT, static_cast<int64_t>(2));
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; "
        "dimensions for image case are (N x C x H x W), "
        "where N is the batch size, C is the number of "
        "channels, and H and W are the height and the width "
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(GlobalLpPool, 22, OpSchema().FillUsing(GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool")));

static const char* BatchNormalization_ver15_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    15,
    OpSchema()
        .NumOutputs({1, 3})
        .SetDoc(BatchNormalization_ver15_doc + GenerateOptionalArgumentsDoc())
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "momentum",
            "Factor used in computing the running mean and variance."
            "e.g., running_mean = running_mean * momentum + mean * (1 - momentum).",
            AttributeProto::FLOAT,
            0.9f)
        .Attr(
            "training_mode",
            "If set to true, it indicates BatchNormalization is being used for training, and outputs 1 "
            "and 2 are to be computed.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "X",
            "Input data tensor from the previous operator; "
            "dimensions are in the form of (N x C x D1 x D2 ... Dn), "
            "where N is the batch size, C is the number of channels. "
            "Statistics are computed for every channel of C over N and D1 to Dn dimensions. "
            "For image data, input dimensions become (N x C x H x W). "
            "The op also accepts single dimension input of size N in which case C is assumed to be 1",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(1, "scale", "Scale tensor of shape (C).", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(2, "B", "Bias tensor of shape (C).", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            3,
            "input_mean",
            "running (training) or estimated (testing) mean tensor of shape (C).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            4,
            "input_var",
            "running (training) or estimated (testing) variance tensor of shape (C).",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "The output tensor of the same shape as X",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            1,
            "running_mean",
            "The running mean after the BatchNormalization operator.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            2,
            "running_var",
            "The running variance after the BatchNormalization operator. This op uses the population size (N) for "
            "calculating variance, and not the sample size N-1.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain scale and bias types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain mean and variance types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // Inputs 1 to 4 must be of rank 1.
          checkInputRank(ctx, 1, 1);
          checkInputRank(ctx, 2, 1);
          checkInputRank(ctx, 3, 1);
          checkInputRank(ctx, 4, 1);

          Dim num_channels;

          if (hasInputShape(ctx, 0)) {
            if (getInputShape(ctx, 0).dim_size() > 1)
              unifyInputDim(ctx, 0, 1, num_channels);
            else
              unifyDim(num_channels, 1);
          }

          unifyInputDim(ctx, 1, 0, num_channels);
          unifyInputDim(ctx, 2, 0, num_channels);
          unifyInputDim(ctx, 3, 0, num_channels);
          unifyInputDim(ctx, 4, 0, num_channels);

          if (ctx.getAttribute("training_mode") && static_cast<int>(ctx.getAttribute("training_mode")->i()) != 0) {
            if (ctx.getNumOutputs() != 3)
              fail_shape_inference("This number of op outputs should be 3 when Training_mode = True, but it is not.");
          } else {
            if (ctx.getNumOutputs() != 1)
              fail_shape_inference("This number of op outputs should be 1 when Training_mode = False, but it is not.");
          }

          if (ctx.getNumOutputs() > 1) {
            TensorShapeProto outputs_shape;
            *outputs_shape.add_dim() = num_channels; // channel

            propagateElemTypeFromInputToOutput(ctx, 3, 1);
            updateOutputShape(ctx, 1, outputs_shape);

            if (ctx.getNumOutputs() > 2) {
              propagateElemTypeFromInputToOutput(ctx, 4, 2);
              updateOutputShape(ctx, 2, outputs_shape);
            }
          }
        }));

static const char* InstanceNormalization_ver22_doc = R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    22,
    OpSchema()
        .SetDoc(InstanceNormalization_ver22_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Input(
            0,
            "input",
            "Input data tensor from the previous operator; "
            "dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "scale",
            "The input 1-dimensional scale tensor of size C.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "B",
            "The input 1-dimensional bias tensor of size C.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "The output tensor of the same shape as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static const char* LpNormalization_ver22_doc = R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    22,
    OpSchema()
        .Input(0, "input", "Input matrix", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Matrix after normalization", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.")
        .SetDoc(LpNormalization_ver22_doc)
        .Attr(
            "axis",
            "The axis on which to apply normalization, -1 mean last axis.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "p",
            "The order of the normalization, only 1 or 2 are supported.",
            AttributeProto::INT,
            static_cast<int64_t>(2))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static const char* Dropout_ver22_doc = R"DOC(
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    22,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(Dropout_ver22_doc) + GenerateOptionalArgumentsDoc()))
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "data", "The input data as Tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "ratio",
            "The ratio of random dropout, with value in [0, 1). If set to 0, "
            "the output would be a simple copy of the input. "
            "If it's non-zero, output will be a random dropout of the scaled input, which is typically "
            "the case during training. It is an optional value, if not specified it will default to 0.5.",
            "T1",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "training_mode",
            "If set to true then it indicates dropout is being used for training. It is an optional value hence unless "
            "specified explicitly, it is false. If it is false, ratio is ignored and the operation mimics inference mode where "
            "nothing will be dropped from the input data and if mask is requested as output it will contain all ones.",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "The output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(1, "mask", "The output mask.", "T2", OpSchema::Optional, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_float_types_ir10(), "Constrain input and output types to float tensors.")
        .TypeConstraint("T1", OpSchema::all_float_types_ir10(), "Constrain input 'ratio' types to float tensors.")
        .TypeConstraint("T2", {"tensor(bool)"}, "Constrain output 'mask' types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }

          if (ctx.getNumInputs() > 1 && hasInputShape(ctx, 1)) {
            auto& ratio_input_shape = getInputShape(ctx, 1);
            if (static_cast<int>(ratio_input_shape.dim_size()) != 0) {
              fail_shape_inference("Ratio of Dropout must be a scalar.");
            }
          }

          if (ctx.getNumInputs() > 2 && hasInputShape(ctx, 2)) {
            auto& training_mode_input_shape = getInputShape(ctx, 2);
            if (static_cast<int>(training_mode_input_shape.dim_size()) != 0) {
              fail_shape_inference("training_mode of Dropout must be a scalar.");
            }
          }

          if (ctx.getNumOutputs() == 2) {
            updateOutputElemType(ctx, 1, TensorProto::BOOL);
            if (hasNInputShapes(ctx, 1)) {
              propagateShapeFromInputToOutput(ctx, 0, 1);
            }
          }
        }));

static const char* Shrink_ver9_doc = R"DOC(
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Shrink,
    9,
    OpSchema()
        .SetDoc(Shrink_ver9_doc)
        .Attr("lambd", "The lambd value for the Shrink formulation. Default is 0.5.", AttributeProto::FLOAT, 0.5f)
        .Attr("bias", "The bias value added to output. Default is 0.", AttributeProto::FLOAT, 0.0f)
        .Input(0, "input", "The input data as Tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "The output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .FunctionBody(
            R"ONNX(
          {
            Lambd = Constant <value_float: float = @lambd>()
            LambdCast = CastLike (Lambd, input)
            Bias = Constant <value_float: float = @bias>()
            BiasCast = CastLike (Bias, input)
            Zero = Constant <value = float {0.0}>()
            ZeroCast = CastLike (Zero, input)
            NegLmbda = Neg (LambdCast)
            InputLessThanNegLambda = Less (input, NegLmbda)
            InputAddBias = Add (input, BiasCast)
            InputSubBias = Sub (input, BiasCast)
            LambdaLessThanInput = Less (LambdCast, input)
            InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias, ZeroCast)
            output = Where(InputLessThanNegLambda, InputAddBias, InputSubBiasOrZero)
          }
        )ONNX",
            18));

static const char* Flatten_ver24_doc = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    24,
    OpSchema()
        .SetDoc(Flatten_ver24_doc)
        .Input(0, "input", "A tensor of rank >= axis.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "A 2D tensor with the contents of the input tensor, "
            "with input dimensions up to axis flattened to the outer dimension "
            "of the output and remaining input dimensions flattened into the inner "
            "dimension of the output.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir12(),
            "Constrain input and output to all tensor types up to IRv12.")
        .Attr(
            "axis",
            "Indicate up to which input dimensions "
            "(exclusive) should be flattened to the outer dimension of the output. "
            "The value for axis must be in the range [-r, r], where r is the rank of the input tensor. "
            "Negative value means counting dimensions from the back. "
            "When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n), "
            "where the shape of the input tensor is (d_0, d_1, ... d_n). ",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasInputShape(ctx, 0))
            return;
          auto& input_shape = getInputShape(ctx, 0);
          int rank = static_cast<int>(input_shape.dim_size());
          int axis = static_cast<int>(getAttribute(ctx, "axis", 1));
          if (axis < 0) {
            axis += rank;
          }
          if (axis > rank || axis < 0) {
            fail_shape_inference("Invalid value(", axis, ") for attribute 'axis'");
          }
          // TODO: is the operation defined for input-rank < 2?
          updateOutputShape(ctx, 0, {multiplyDims(input_shape, 0, axis), multiplyDims(input_shape, axis, rank)});
        }));

static const char* LRN_ver13_doc = R"DOC(
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
of shape `(N x C x D1 x D2, ..., Dk)`, its region is
`{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

`square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

`Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    13,
    OpSchema()
        .Attr("size", "The number of channels to sum over", AttributeProto::INT)
        .Attr("alpha", "Scaling parameter.", AttributeProto::FLOAT, 0.0001f)
        .Attr("beta", "The exponent.", AttributeProto::FLOAT, 0.75f)
        .Attr("bias", "", AttributeProto::FLOAT, 1.0f)
        .Input(
            0,
            "X",
            "Input data tensor from the previous operator; "
            "dimensions for image case are (N x C x H x W), "
            "where N is the batch size, C is the number of "
            "channels, and H and W are the height and the "
            "width of the data. For non image case, the "
            "dimensions are in the form of "
            "(N x C x D1 x D2 ... Dn), where N is the batch "
            "size. Optionally, if dimension denotation is "
            "in effect, the operation expects the input "
            "data tensor to arrive with the dimension denotation "
            "of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "Output tensor, which has the shape and type as input tensor",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output "
            " types to float tensors.")
        .SetDoc(LRN_ver13_doc)
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* TfIdfVectorizer_ver9_doc = R"DOC(
This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

The output vector (denoted by Y) stores the count of each n-gram;
Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    TfIdfVectorizer,
    9,
    OpSchema()
        .Input(0, "X", "Input for n-gram extraction", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "Ngram results", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(string)", "tensor(int32)", "tensor(int64)"},
            "Input is ether string UTF-8 or int32/int64")
        .TypeConstraint("T1", {"tensor(float)"}, "1-D tensor of floats")
        .Attr(
            "max_gram_length",
            "Maximum n-gram length. If this value is 3, 3-grams will be used to generate the output.",
            AttributeProto::INT)
        .Attr(
            "min_gram_length",
            "Minimum n-gram length. If this value is 2 and max_gram_length is 3, output may contain counts of 2-grams and 3-grams.",
            AttributeProto::INT)
        .Attr(
            "max_skip_count",
            "Maximum number of items (integers/strings) to be skipped when constructing an n-gram from X. "
            "If max_skip_count=1, min_gram_length=2, max_gram_length=3, this operator may generate 2-grams "
            "with skip_count=0 and skip_count=1, and 3-grams with skip_count=0 and skip_count=1",
            AttributeProto::INT)
        .Attr(
            "pool_strings",
            "List of strings n-grams learned from the training set. Either this or pool_int64s attributes must be present but not both. "
            "It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. "
            "The i-th element in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "pool_int64s",
            "List of int64 n-grams learned from the training set. Either this or pool_strings attributes must be present but not both. "
            "It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. "
            "The i-th element in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "ngram_counts",
            "The starting indexes of 1-grams, 2-grams, and so on in pool. "
            "It is useful when determining the boundary between two consecutive collections of n-grams. "
            "For example, if ngram_counts is [0, 17, 36], the first index (zero-based) of 1-gram/2-gram/3-gram "
            "in pool are 0/17/36. This format is essentially identical to CSR (or CSC) sparse matrix format, "
            "and we choose to use this due to its popularity.",
            AttributeProto::INTS)
        .Attr(
            "ngram_indexes",
            "list of int64s (type: AttributeProto::INTS). This list is parallel to the specified 'pool_*' attribute. "
            "The i-th element in ngram_indexes indicate the coordinate of the i-th n-gram in the output tensor.",
            AttributeProto::INTS)
        .Attr(
            "weights",
            "list of floats. This attribute stores the weight of each n-gram in pool. The i-th element in weights "
            "is the weight of the i-th n-gram in pool. Its length equals to the size of ngram_indexes. "
            "By default, weights is an all-one tensor.This attribute is used when mode is \"IDF\" or \"TFIDF\" "
            "to scale the associated word counts.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "mode",
            "The weighting criteria. It can be one of \"TF\" (term frequency), "
            "\"IDF\" (inverse document frequency), and \"TFIDF\" (the combination of TF and IDF)",
            AttributeProto::STRING)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::FLOAT);

          if (hasInputShape(ctx, 0)) {
            std::vector<int64_t> ngram_indexes;
            getRepeatedAttribute(ctx, "ngram_indexes", ngram_indexes);
            if (ngram_indexes.empty() ||
                !std::all_of(ngram_indexes.cbegin(), ngram_indexes.cend(), [](int64_t i) { return i >= 0; })) {
              fail_shape_inference("ngram_indexes must be non-empty with no negative values");
            }

            auto greatest_hit = std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend());
            auto max_last_axis = *greatest_hit + 1;

            TensorShapeProto output_shape;
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            auto dim_size = input_shape.dim_size();
            if (dim_size == 1) {
              output_shape.add_dim()->set_dim_value(max_last_axis);
            } else if (dim_size == 2) {
              *output_shape.add_dim() = input_shape.dim(0);
              output_shape.add_dim()->set_dim_value(max_last_axis);
            } else {
              fail_shape_inference("Input tensor must have rank 1 or 2");
            }
            updateOutputShape(ctx, 0, output_shape);
          }
        })
        .SetDoc(TfIdfVectorizer_ver9_doc));

static const char* mvn_ver13_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
)DOC";

static const std::vector<int64_t> mvn_default_axes = {0, 2, 3};

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    13,
    OpSchema()
        .SetDoc(mvn_ver13_doc)
        .Input(0, "X", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Output tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr(
            "axes",
            "A list of integers, along which to reduce. The default is to "
            "calculate along axes [0,2,3] for calculating mean and variance "
            "along each channel. Two variables with the same C-coordinate "
            "are associated with the same mean and variance.",
            AttributeProto::INTS,
            mvn_default_axes)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input and output types to all numeric tensors.")
        .FunctionBody(R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          X_RM = ReduceMean <axes : ints = @axes> (X)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean <axes : ints = @axes> (X_squared)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX")
        .FunctionBody(
            R"ONNX(
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          axes = Constant <value_ints: ints = @axes>()
          X_RM = ReduceMean (X, axes)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean (X_squared, axes)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        )ONNX",
            18));

static void col2imShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // All inputs shapes are required
  if (!hasNInputShapes(ctx, 3)) {
    return;
  }

  // We assume image_shape has correct spatial dimensions for next validations
  // An alternative is get the the number of spatial dimensions as an input argument
  Dim n_input_dims;
  unifyInputDim(ctx, 1, 0, n_input_dims);

  unifyInputDim(ctx, 2, 0, n_input_dims);
  checkInputRank(ctx, 1, 1);
  checkInputRank(ctx, 2, 1);
  std::vector<int64_t> image_shape = {};
  const TensorProto* image_shape_data = ctx.getInputData(1);
  if (image_shape_data) {
    image_shape = ParseData<int64_t>(image_shape_data);
    unifyDim(n_input_dims, image_shape.size());
  }

  std::vector<int64_t> pads = {};
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() % 2) {
      fail_shape_inference("Attribute pads must have an even size");
    }
    unifyDim(n_input_dims, pads.size() / 2);
  }

  std::vector<int64_t> dilations = {};
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    unifyDim(n_input_dims, dilations.size());
  }

  std::vector<int64_t> strides = {};
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    unifyDim(n_input_dims, strides.size());
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() != 3) {
    fail_shape_inference("input must have rank 3.");
  }

  std::vector<int64_t> block_shape = {};
  const TensorProto* block_shape_data = ctx.getInputData(2);
  if (block_shape_data) {
    block_shape = ParseData<int64_t>(block_shape_data);
    unifyDim(n_input_dims, block_shape.size());
  }
  unifyInputDim(ctx, 2, 0, n_input_dims);

  int block_shape_size = 0;
  if (static_cast<int>(block_shape.size()) > 0) {
    block_shape_size = 1;
    for (const auto& dim : block_shape) {
      block_shape_size *= dim;
    }
  }
  // If we haven't inferred the number of image dimensions, we can't set inferred shape.
  if (!n_input_dims.has_dim_value()) {
    return;
  }

  // Final shape will be (N, C, dim_1, ..., dim_N)
  auto final_image_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  // Dimensions N and C are always present
  Dim N, C;
  if (ctx.getInputType(0)->tensor_type().shape().dim(0).has_dim_value()) {
    N = input_shape.dim(0); // Otherwise, N is unknown.
  }
  *final_image_shape->add_dim() = N;

  if (block_shape_size > 0) {
    C = input_shape.dim(1) / block_shape_size; // Otherwise, C is unknown.
  }
  *final_image_shape->add_dim() = C;

  // Image dimensions are dynamic
  for (auto i = 0; i < n_input_dims.dim_value(); ++i) {
    Dim image_dim_i;
    if (!image_shape.empty()) {
      image_dim_i.set_dim_value(image_shape[i]); // Otherwise, spatial dimensions are unknown
    }
    *final_image_shape->add_dim() = image_dim_i;
  }
  return;
}

static const char* Col2Im_ver18_doc = R"DOC(
The operator rearranges column blocks back into a multidimensional image

Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
but it only supports *batched* multi-dimensional image tensors.
Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

NOTE:
  Although specifying image_shape looks redundant because it could be calculated from
  convolution formulas, it is required as input for more advanced scenarios as explained
  at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Col2Im,
    18,
    OpSchema()
        .Attr(
            "dilations",
            "1-dimensional tensor with dilation value along each spatial axis of the image. "
            "If not present, the dilation defaults to 1 along each spatial axis of the image.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "pads",
            "1-dimensional tensor with padding value for the beginning and ending along each spatial axis, "
            "it can take any value greater than or equal to 0. "
            "The value represent the number of pixels added to the beginning "
            "and end part of the corresponding axis. `pads` format should be as follow "
            "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin is the number of pixels "
            "added at the beginning of axis `i` and xi_end is the number of pixels added at the end of axis `i`. "
            "If not present, the padding defaults to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "strides",
            "1-dimensional tensor with stride value along each spatial axis. "
            "If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .SetDoc(Col2Im_ver18_doc)
        .Input(
            0,
            "input",
            "Input data tensor to be rearranged from column blocks back into an image."
            " This is a 3-dimensional tensor containing [N, C * n-ary-product(block_shape), L],"
            " where N is batch dimension, C is image channel dimension and L is number of blocks."
            "The blocks are enumerated in increasing lexicographic-order of their indices."
            "For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks,"
            " enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, 1), block(1, 2).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "image_shape",
            "The shape of the spatial dimensions of the image after rearranging the column blocks."
            "This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img] "
            " for a 2-D image or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "block_shape",
            "The shape of the block to apply on the input."
            "This is a 1-dimensional tensor of size of at least 2, containing the value [H_block, W_block] "
            " for a 2-D image or [dim_b1, dim_b2, ..., dim_bN] for a N-D block."
            "This is the block-shape before dilation is applied to it.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by rearranging blocks into an image.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir4(),
            "Constrain input and output types to all numeric tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { col2imShapeInference(ctx); }));

static const char* LayerNormalization_ver17_doc = R"DOC(
      This is layer normalization defined in ONNX as function.
      The overall computation can be split into two stages.
      The first stage is standardization, which makes the
      normalized elements have zero mean and unit variances.
      The computation required by standardization can be
      described by the following equations.
      ```
      Mean = ReduceMean<axes=normalized_axes>(X)
      D = Sub(X, Mean)
      DD = Mul(D, D)
      Var = ReduceMean<axes=normalized_axes>(DD)
      VarEps = Add(Var, epsilon)
      StdDev = Sqrt(VarEps)
      InvStdDev = Reciprocal(StdDev)
      Normalized = Mul(D, InvStdDev)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`.
      The variables `Var` and `StdDev` stand for variance and
      standard deviation, respectively. The second output is
      `Mean` and the last one is `InvStdDev`.
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales and shifts the outcome of the
      first stage using
      ```
      NormalizedScaled = Mul(Normalized, Scale)
      Y = Add(NormalizedScaled, B)
      ```
      The second stage doesn't depends on `stash_type`.
      All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
      The same variable (i.e., input, output, and attribute) uses
      the same name in the equations above and this operator's definition.
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
      (tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
      for more details please check [the doc](Broadcasting.md).
)DOC";

static bool BuildContextDependentFunctionBodyLayerNormalization(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto,
    int sinceVersion) {
  ONNX_ASSERT(sinceVersion == 17 || sinceVersion == 18)
  // LayerNormalization <axis, epsilon, stash_type> (X, Scale, B) => (Y, Mean?, InvStdDev?)
  auto* tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  int64_t T = tp->tensor_type().elem_type();

  auto type_attr = ctx.getAttribute("stash_type");
  int64_t U =
      (type_attr != nullptr) ? type_attr->i() : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) && (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16))
    return false; // Error

  auto* axis_attr = ctx.getAttribute("axis");
  int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
  auto* epsilon_attr = ctx.getAttribute("epsilon");
  float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

  auto mktensor = [](int64_t val) -> ONNX_NAMESPACE::TensorProto {
    auto tp = ONNX_NAMESPACE::ToTensor(std::vector<int64_t>{val});
    tp.add_dims(1);
    return tp;
  };
  // The treatment of "axis" is different in "LayerNormalization" and in Reduction operations.
  // This complicates the function definition, requiring reshaping inputs/outputs.
  // Input X shape: [d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]
  // This is treated as a 2D shape [d[0] * ... * d[axis-1], d[axis] * ... * d[rank-1]]
  // Normalization is applied to the second dimension.
  // Output Y has same shape as X
  // Outputs Mean and InvStdDev have shape: [d[0], ..., d[axis-1], 1, ..., 1]
  FunctionBuilder builder(functionProto);
  builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
      .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
      .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
      .Add("Rank = Size (XShape)") // rank of input tensor: scalar
      .Add("Zero1D = Constant()", "value", mktensor(0)) // [0] : 1D tensor
      .Add("Axis1D = Constant()", "value", mktensor(axis)) // [axis] : 1D tensor
      .Add("PrefixShape = Slice (XShape, Zero1D, Axis1D)") // [d[0], ..., d[axis-1]]
      .Add(
          axis >= 0 // number of axes that are reduced =
              ? "NumReducedAxes = Sub (Rank, Axis1D)" // [rank - axis]: 1D tensor
              : "NumReducedAxes = Neg (Axis1D)") // [-axis] : 1D tensor
      .Add(
          "SuffixShape = ConstantOfShape (NumReducedAxes)",
          "value",
          mktensor(1)) // [1, ..., 1] for reduced axes
      .Add("ReducedShape = Concat <axis = 0> (PrefixShape, SuffixShape)") // [d[0], ..., d[axis-1], 1, ..., 1]
      .Add("X2D = Flatten (X)", "axis", axis)
      .Add("XU = Cast (X2D)", "to", U);
  if (sinceVersion == 17) {
    builder.Add("Mean2D = ReduceMean <axes = [1]> (XU)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean <axes = [1]> (Square)");
  } else if (sinceVersion == 18) {
    builder.Add("Axes_1 = Constant()", "value", mktensor(1))
        .Add("Mean2D = ReduceMean (XU, Axes_1)")
        .Add("Square = Mul (XU, XU)")
        .Add("MeanOfSquare = ReduceMean (Square, Axes_1)");
  }
  builder.Add("SquareOfMean = Mul (Mean2D, Mean2D)")
      .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
      .Add("VarPlusEpsilon = Add (Var, Epsilon)")
      .Add("StdDev = Sqrt (VarPlusEpsilon)")
      .Add("Deviation = Sub (XU, Mean2D)")
      .Add("Normalized = Div (Deviation, StdDev)")
      .Add("NormalizedT = Cast (Normalized)", "to", T)
      .Add("Scale2D = Flatten <axis = 0> (Scale)")
      .Add("Scaled = Mul (NormalizedT, Scale2D)");
  if (ctx.hasInput(2)) {
    builder.Add("B2D = Flatten <axis=0> (B)");
    builder.Add("Biased = Add (Scaled, B2D)");
  } else {
    builder.Add("Biased = Identity (Scaled)");
  }
  builder.Add("Y = Reshape (Biased, XShape)");
  builder.Add("InvStdDev2D = Reciprocal (StdDev)");
  if (ctx.hasOutput(1))
    builder.Add("Mean = Reshape (Mean2D, ReducedShape)");
  if (ctx.hasOutput(2))
    builder.Add("InvStdDev = Reshape (InvStdDev2D, ReducedShape)");

  schema.BuildFunction(functionProto);
  return true;
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer17(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 17);
}

static bool BuildContextDependentFunctionBodyLayerNormalizationVer18(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  return BuildContextDependentFunctionBodyLayerNormalization(ctx, schema, functionProto, 18);
}

ONNX_OPERATOR_SET_SCHEMA(
    LayerNormalization,
    17,
    OpSchema()
        .SetDoc(LayerNormalization_ver17_doc)
        .Attr(
            "axis",
            "The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). "
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "stash_type",
            "Type of Mean and InvStdDev. This also specifies stage one's computation precision.",
            AttributeProto::INT,
            static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
        .AllowUncheckedAttributes()
        .Input(0, "X", "Tensor to be normalized.", "T")
        .Input(1, "Scale", "Scale tensor.", "T")
        .Input(2, "B", "Bias tensor.", "T", OpSchema::Optional)
        .Output(0, "Y", "Normalized tensor.", "T")
        .Output(1, "Mean", "Saved mean used during training to speed up gradient computation", "U", OpSchema::Optional)
        .Output(
            2,
            "InvStdDev",
            "Saved inverse standard deviation used during training to speed up gradient computation.",
            "U",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input types and output Y type to float tensors.")
        .TypeConstraint("U", {"tensor(float)", "tensor(bfloat16)"}, "Type of Mean and InvStdDev tensors.")
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer17, 17)
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyLayerNormalizationVer18, 18)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          auto stash_type = static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          auto stash_type_proto = ctx.getAttribute("stash_type");
          if (stash_type_proto) {
            stash_type = stash_type_proto->i();
          }
          if (ctx.getNumOutputs() > 1) {
            auto output_type = ctx.getOutputType(1);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (ctx.getNumOutputs() > 2) {
            auto output_type = ctx.getOutputType(2);
            output_type->mutable_tensor_type()->set_elem_type(static_cast<int32_t>(stash_type));
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto& input_shape = getInputShape(ctx, 0);
          int64_t input_ndim = input_shape.dim_size();
          int64_t axis = -1;
          auto axis_proto = ctx.getAttribute("axis");
          if (axis_proto) {
            axis = axis_proto->i();
          }
          if (axis < 0) {
            // Convert negative axis value to equivalent
            // positive value.
            axis += input_ndim;
          }
          if (axis < 0) {
            fail_shape_inference(
                "Unexpected axis value (",
                axis,
                ") rank of first input is ",
                input_ndim,
                " in ",
                ctx.getDisplayName(),
                ".");
          }
          if (ctx.getNumOutputs() > 1) {
            auto mean_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
            mean_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              mean_shape->mutable_dim(d)->set_dim_value(1);
          }

          if (ctx.getNumOutputs() > 2) {
            auto inv_std_dev_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
            inv_std_dev_shape->CopyFrom(input_shape);
            for (int d = static_cast<int>(axis); d < input_ndim; ++d)
              inv_std_dev_shape->mutable_dim(d)->set_dim_value(1);
          }
        }));

static const char* GroupNormalization_ver21_doc = R"DOC(
A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each channel. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

The overall computation has two stages: the first stage normalizes the elements to
have zero mean and unit variance for each instance in each group, and the second
stage scales and shifts the results of the first stage. The floating-point precision
used in the first stage is determined by the `stash_type` attribute. For example,
if `stash_type` is 1, the operator casts all input variables to 32-bit float,
performs the computation, and finally casts the normalized results back to the
original type of `X`. The second stage does not depend on `stash_type`.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GroupNormalization,
    21,
    OpSchema()
        .SetDoc(GroupNormalization_ver21_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "num_groups",
            "The number of groups of channels. It should be a divisor of the number of channels `C`.",
            AttributeProto::INT,
            true)
        .Attr(
            "stash_type",
            "The floating-point precision used in stage one of the computation.",
            AttributeProto::INT,
            static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
        .Input(
            0,
            "X",
            "Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, "
            "`C` is the number of channels, and `H` and `W` are the height and width of the data. Statistics are "
            "computed for every group of channels over `C`, `H`, and `W`. For non-image cases, the dimensions are "
            "in the form of `(N x C x D1 x D2 ... Dn)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(1, "scale", "Scale tensor of shape `(C)`.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(2, "bias", "Bias tensor of shape `(C)`.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "Y",
            "The output tensor of the same shape as `X`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_float_types_ir4(), "Constrain input and output types to float tensors.")
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
              // GroupNormalization <epsilon, num_groups> (X, scale, bias) => (Y)
              auto* tp = ctx.getInputType(0);
              if ((tp == nullptr) || (!tp->has_tensor_type()))
                return false;
              int64_t in_type = tp->tensor_type().elem_type();

              auto* epsilon_attr = ctx.getAttribute("epsilon");
              float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;
              auto* num_groups_attr = ctx.getAttribute("num_groups");
              if (num_groups_attr == nullptr)
                return false;
              int64_t num_groups = num_groups_attr->i();

              auto stash_type_attr = ctx.getAttribute("stash_type");
              int64_t stash_type = (stash_type_attr != nullptr)
                  ? stash_type_attr->i()
                  : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              if ((stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
                  (stash_type != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
                return false; // Error

              FunctionBuilder builder(functionProto);
              builder.Const1D("FloatEpsilon", epsilon)
                  .Add("Epsilon = Cast (FloatEpsilon)", "to", stash_type)
                  .Add("XU = Cast (X)", "to", stash_type)
                  .Add("XShape = Shape (XU)") // shape of input tensor: 1D tensor
                  .Add("C = Shape <start = 1, end = 2> (X)")
                  .Const1D("NumGroups", num_groups)
                  .Add("GroupSize = Div (C, NumGroups)")
                  .Add("N = Shape <start = 0, end = 1> (X)") // batch size
                  .Add("InstanceShape = Shape <start = 2> (X)") // data instance shape

                  // NewShape = [N, num_groups, group_size, H, W, (...)]
                  .Add("NewShape = Concat <axis = 0> (N, NumGroups, GroupSize, InstanceShape)")
                  .Add("XReshaped = Reshape (XU, NewShape)")

                  // Flatten into 3D tensor: [N, num_groups, group_size x H x W (x ...)]
                  .Add("Shape3D = Constant <value_ints = [0, 0, -1]> ()")
                  .Add("X3D = Reshape (XReshaped, Shape3D)")

                  // Calculate statistics
                  .Const1D("Axes2", (int64_t)2)
                  .Add("Mean = ReduceMean (X3D, Axes2)")
                  .Add("Square = Mul (X3D, X3D)")
                  .Add("MeanOfSquare = ReduceMean (Square, Axes2)")
                  .Add("SquareOfMean = Mul (Mean, Mean)")
                  .Add("Var = Sub (MeanOfSquare, SquareOfMean)")
                  .Add("VarPlusEpsilon = Add (Var, Epsilon)")
                  .Add("StdDev = Sqrt (VarPlusEpsilon)")
                  .Add("Deviation = Sub (X3D, Mean)")
                  .Add("NormalizedU = Div (Deviation, StdDev)")

                  // Reshape to [N, C, H x W (x ...)] and cast to original type
                  .Add("NormalizedOriginalShape = Reshape (NormalizedU, XShape)")
                  .Add("NormalizedNC = Reshape (NormalizedOriginalShape, Shape3D)")
                  .Add("NormalizedT = Cast (NormalizedNC)", "to", in_type)

                  // Reshape scale and bias to [1, C, 1] for broadcasting
                  .Add("ScaleShape = Constant <value_ints = [1, -1, 1]> ()")
                  .Add("ScaleT = Cast (scale)", "to", in_type)
                  .Add("BiasT = Cast (bias)", "to", in_type)
                  .Add("ScaleReshaped = Reshape (ScaleT, ScaleShape)")
                  .Add("BiasReshaped = Reshape (BiasT, ScaleShape)")

                  // Calculate scaled and biased output
                  .Add("Scaled = Mul (ScaleReshaped, NormalizedT)")
                  .Add("Biased = Add (Scaled, BiasReshaped)")
                  .Add("Y = Reshape (Biased, XShape)");

              schema.BuildFunction(functionProto);
              return true;
            }));

static const char* RMSNormalization_ver23_doc = R"DOC(
      This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.org/pdf/1910.07467.
      The overall computation can be split into two stages. The root mean squared norm is taken over the last D dimensions,
      where D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5) (a 2-dimensional shape),
      the rms norm is computed over the last 2 dimensions of the input. The computation required by standardization can be
      described by the following equations.
      ```
      XSquared = Mul(X, X)
      XSquaredMean = ReduceMean<axes=normalized_axes>(XSquared)
      MeanSquareEpsilon = Add(XSquaredMean, epsilon)
      RMS = Sqrt(MeanSquareEpsilon)
      Normalized = Div(X, RMS)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`. The variables `RMS` stand for root mean square,
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales the outcome of the first stage using:
      ```
      Y= Mul(Normalized, Scale)
      ```
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `RMS` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
      (`Scale` should be unidirectional broadcastable to tensor `X`);
      for more details please check [the doc](Broadcasting.md).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RMSNormalization,
    23,
    OpSchema()
        .SetDoc(RMSNormalization_ver23_doc)
        .Attr(
            "axis",
            "The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). "
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr("epsilon", "The epsilon value to use to avoid division by zero.", AttributeProto::FLOAT, 1e-5f)
        .Attr(
            "stash_type",
            "The floating-point precision used in stage one of the computation.",
            AttributeProto::INT,
            static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))
        .Input(
            0,
            "X",
            "The input tensor to be normalized. "
            "In general, the shape is (D1, D2, ... , Dn) for n-dimensional data, where "
            "the root mean squared norm is taken over the last D dimensions, D is determined by the axis attribute.",
            "T")
        .Input(1, "scale", "Scale tensor. Scale tensor shape should be broadcastable to the normalized shape.", "V")
        .Output(0, "Y", "Output data tensor. Same shape as X", "V")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain input X type to float tensors.")
        .TypeConstraint(
            "V",
            {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
            "Constrain output Y and scale type to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          int64_t input_ndim = input_shape.dim_size();
          int64_t axis = -1;
          auto axis_proto = ctx.getAttribute("axis");
          if (axis_proto) {
            axis = axis_proto->i();
          }
          if (axis < 0) {
            // Convert negative axis value to equivalent
            // positive value.
            axis += input_ndim;
          }
          if (axis < 0) {
            fail_shape_inference(
                "Unexpected axis value (",
                axis,
                ") rank of first input is ",
                input_ndim,
                " in ",
                ctx.getDisplayName(),
                ".");
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // RMSNormalization <axis, epsilon, stash_type> (X, Scale) => (Y)
          auto* tp = ctx.getInputType(0);
          if ((tp == nullptr) || (!tp->has_tensor_type()))
            return false;
          int64_t T = tp->tensor_type().elem_type();

          auto type_attr = ctx.getAttribute("stash_type");
          int64_t U = (type_attr != nullptr) ? type_attr->i()
                                             : static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
          if ((U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (U != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (U != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) && (U != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
            return false; // Error

          auto* axis_attr = ctx.getAttribute("axis");
          int64_t axis = (axis_attr != nullptr) ? axis_attr->i() : -1;
          auto* epsilon_attr = ctx.getAttribute("epsilon");
          float epsilon = (epsilon_attr != nullptr) ? epsilon_attr->f() : 1e-5f;

          FunctionBuilder builder(functionProto);
          builder.Const("FloatEpsilon", ToTensor<float>(epsilon))
              .Add("Epsilon = Cast (FloatEpsilon)", "to", U)
              .Add("XShape = Shape (X)") // shape of input tensor: 1D tensor
              .Add("Rank = Size (XShape)") // rank of input tensor: scalar
              .Const("Axis", axis) // axis : scalar
              .Add(
                  axis >= 0 // number of axes that are reduced =
                      ? "PosAxis = Identity (Axis)" // axis: scalar
                      : "PosAxis = Add (Rank, Axis)") // rank + axis : scalar
              .Const("One", (int64_t)1)
              .Add("ReduceAxes = Range(PosAxis, Rank, One)")
              .Add("XU = Cast (X)", "to", U);
          builder.Add("XSquared = Mul (XU, XU)")
              .Add("XSquaredMean = ReduceMean (XSquared, ReduceAxes)")
              .Add("MeanSquareEpsilon = Add (XSquaredMean, Epsilon)")
              .Add("RMS = Sqrt (MeanSquareEpsilon)")
              .Add("Normalized = Div (XU, RMS)")
              .Add("NormalizedT = Cast (Normalized)", "to", T);
          builder.Add("Y = Mul (NormalizedT, scale)");

          schema.BuildFunction(functionProto);
          return true;
        }));

static const char* RotaryEmbedding_ver23_doc = R"DOC(
RotaryEmbedding is the implementation of rotary positional embeddings (RoPE) based on the paper https://arxiv.org/pdf/2104.09864.
The key advantage of RoPE is that it allows the model to understand both the absolute position of a token and the relative distances
between tokens. This is achieved through a rotational mechanism where the extent of rotation is computed based on the token's absolute position (position_ids).

The rotational mechanism is defined by sine and cosine functions that are used to represent the rotation angles.
For each token in the sequence, its positional embedding is computed by rotating its embedding vector. This is done by splitting the
embedding vector either into two halves or interleaving every alternate token and applying the rotation matrix to each half of the embedding vector.
The rotation matrix is parameterized by the token's position in the sequence. The rotated halves of the embedding vector are concatenated
to form the final positional embedding for each token. The rotated positional embeddings are used in the self-attention mechanism.
The rotation ensures that the model captures both absolute and relative positional information.

Rotary embeddings are defined using the following algorithm:

```python
def compute_rotary_embedding(
    input,
    position_ids,
    sin_cache,
    cos_cache,
    interleaved=0,
    rotary_embedding_dim=0,
    num_heads=0,
):
    # First ensure input to be processed has shape [batch_size, seq_len, num_heads, head_size]
    if len(input.shape) == 4:
        input = np.transpose(input, (0, 2, 1, 3))
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    if len(input.shape) == 3:
        hidden_size = input.shape[2]
        assert num_heads != 0
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input = np.reshape(input, new_shape)
    assert len(input.shape) == 4
    head_size = input.shape[3]

    # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    if rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos = cos_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
        sin = sin_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
    else:
        cos = cos_cache
        sin = sin_cache
    cos = cos[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    sin = sin[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    cos = np.expand_dims(cos, axis=2)  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin = np.expand_dims(sin, axis=2)  # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the input in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)

    # Calculate real and imaginary values
    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    # Inserted rotated embeddings back to the original input
    if interleaved:
        # x_rotate[:, :, :, 0::2] = real
        # x_rotate[:, :, :, 1::2] = imag
        real = np.expand_dims(real, axis=-1)
        imag = np.expand_dims(imag, axis=-1)
        x_rotate_concat = np.concatenate((real, imag), axis=-1)
        x_rotate = np.reshape(x_rotate_concat, x_rotate.shape)
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)
    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if len(original_input_shape) == 3:
        output = np.reshape(output, input.shape)
    else:
        output = np.transpose(output, (0, 2, 1, 3))
    return output
```
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RotaryEmbedding,
    23,
    OpSchema()
        .SetDoc(RotaryEmbedding_ver23_doc)
        .Attr(
            "interleaved",
            "Rotate using interleaved pattern. Default value is 0 (False).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "rotary_embedding_dim",
            "Rotary embedding dimension used to apply partial rotary embeddings.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "num_heads",
            "Number of attention heads. Must be provided when input is a 3D tensor. ",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(
            0,
            "X",
            "The input tensor representing the token embeddings. "
            "4D tensor with shape `(batch_size, num_heads, sequence_length, head_size)` or 3D tensor with shape `(batch_size, sequence_length, hidden_size)`. "
            "For cases with a 4D input tensor, `head_size` has to be even. For cases with a 3D input tensor, `num_heads` attribute must be provided and "
            "`hidden_size` must be an even multiple of `num_heads` where `hidden_size = num_heads * head_size`",
            "T")
        .Input(
            1,
            "cos_cache",
            "The cosine values for the rotation. "
            "2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)` "
            "for partial rotation when `position_ids` are provided. 3D tensor with shape `(batch_size, sequence_length, head_size / 2)` "
            "for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are not provided. "
            "`max_position_id_plus_1` is a parameter to the model.",
            "T")
        .Input(
            2,
            "sin_cache",
            "The sine values for the rotation. "
            "2D tensor with shape `(max_position_id_plus_1, head_size / 2)` for full rotation or `(max_position_id_plus_1, rotary_embedding_dim / 2)` "
            "for partial rotation when `position_ids` are provided. 3D tensor with shape `(batch_size, sequence_length, head_size / 2)` "
            "for full rotation or `(batch_size, sequence_length, rotary_embedding_dim / 2)` for partial rotation when `position_ids` are not provided. "
            "`max_position_id_plus_1` is a parameter to the model.",
            "T")
        .Input(
            3,
            "position_ids",
            "The position indices for the tokens. 2D tensor with shape `(batch_size, sequence_length)`",
            "M",
            OpSchema::Optional)
        .Output(0, "Y", "Tensor with same shape as input.", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint("M", {"tensor(int64)"}, "Constrain input and output types to integer tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          propagateShapeFromInputToOutput(ctx, 0, 0);

          // we need at least one input to have a shape for this inference.
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          if ((input_shape.dim_size() < 3) || (input_shape.dim_size() > 4)) {
            fail_shape_inference("Input tensor must have at least 3 and at most 4 dimensions");
          }

          auto* num_heads_attr = ctx.getAttribute("num_heads");
          if ((input_shape.dim_size() == 3) && (num_heads_attr == nullptr)) {
            fail_shape_inference("Input shape is 3D, num_heads attribute must be provided");
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // RotaryEmbedding <scale, interleaved, rotary_embedding_dim, num_heads> (X, position_ids, cos_cache,
          // sin_cache) => Y

          int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          auto* interleaved_attr = ctx.getAttribute("interleaved");
          int64_t interleaved = (interleaved_attr != nullptr) ? interleaved_attr->i() : 0;
          auto* rotary_embedding_dim_attr = ctx.getAttribute("rotary_embedding_dim");
          int64_t rotary_embedding_dim = (rotary_embedding_dim_attr != nullptr) ? rotary_embedding_dim_attr->i() : 0;
          auto* num_heads_attr = ctx.getAttribute("num_heads");
          int64_t num_heads = (num_heads_attr != nullptr) ? num_heads_attr->i() : 0;

          // Ensure that num_heads does not control reshaping of input tensor
          // when input tensor is 4D
          int64_t is_input_4d = 1;
          auto* x_tp = ctx.getInputType(0);
          if ((x_tp == nullptr) || (!x_tp->has_tensor_type()))
            return false;
          if (!(x_tp->tensor_type().has_shape())) {
            return false;
          }
          if (x_tp->tensor_type().shape().dim_size() == 4) {
            is_input_4d = 1;
          } else if (x_tp->tensor_type().shape().dim_size() == 3) {
            is_input_4d = 0;
          } else {
            return false;
          }

          FunctionBuilder builder(functionProto);
          // Set input tensor to the correct shape if input shape is 3D
          // NewShape = [batch_size, sequence_length, num_heads, head_size]

          // Reshape tensor to 4D if input is 3D
          builder.Const1D("Zero1D", (int64_t)0)
              .Const1D("NumHeads", num_heads) // num_heads
              .Const1D("NegOne", (int64_t)(-1)); // head_size, inferred from other dimensions

          if (is_input_4d == 0) {
            builder.Add("NewShape = Concat <axis = 0> (Zero1D, Zero1D, NumHeads, NegOne)")
                .Add("XIn = Reshape (X, NewShape)"); // new shape of input tensor: 4D tensor
          } else {
            builder.Add("XIn = Transpose <perm = [0, 2, 1, 3]> (X)");
          }

          // Rotary embedding dimension is the value along which the input is to be split
          // There are two cases for the rotary embedding dimension:
          // 1. Complete rotation: rotary embedding dimension defaults to head_size, rotary_embedding_dim = cos.shape[3]
          // * 2 or head_size
          // 2. Partial rotation: rotary embedding dimension is provided, rotary_embedding_dim = rotary_embedding_dim

          builder.Add("HeadSize = Shape <start = 3, end = 4> (XIn)");
          if (rotary_embedding_dim > 0) {
            builder.Const1D("RotaryEmbedDim", rotary_embedding_dim);
          } else {
            builder.Add("RotaryEmbedDim = Identity(HeadSize)");
          }
          builder.Const1D("Two1D", (int64_t)2)
              .Add("NoRotateLength = Sub(HeadSize, RotaryEmbedDim)")
              .Add("RotateSplitLengths = Concat <axis = 0> (RotaryEmbedDim, NoRotateLength)");
          // shape of input to rotate = input[:,:,:,:rotary_embedding_dim]
          // shape of input not to rotate = input[:,:,:,rotary_embedding_dim:]
          builder.Add("XToRotate, XNoRotate = Split <axis = -1> (XIn, RotateSplitLengths)");

          // Gather the cos and sine matrices from the respective caches using position ids if provided.
          // Otherwise Gather op functions as an Identity op.
          // Unsqueeze applied to make cos and sin matrices have dimensions that are
          // valid for multiplication with input when is split. For cases where rotary_embedding_dim is provided,
          // slice the matrix values until that index only
          if (ctx.hasInput(3)) {
            builder
                .Add("CosCacheGather = Gather(cos_cache, position_ids)") // shape of cos matrix: [batch_size,
                                                                         // sequence_length, head_size / 2]
                .Add("SinCacheGather = Gather(sin_cache, position_ids)"); // shape of cos matrix: [batch_size,
                                                                          // sequence_length, head_size / 2]
          } else {
            builder
                .Add("CosCacheGather = Identity(cos_cache)") // shape of cos matrix: [batch_size, sequence_length,
                                                             // head_size / 2]
                .Add("SinCacheGather = Identity(sin_cache)"); // shape of cos matrix: [batch_size, sequence_length,
                                                              // head_size / 2]
          }

          builder.Add("RotaryEmbedDimHalf = Div(RotaryEmbedDim, Two1D)")
              .Add("RotaryEmbedDimHalfInt = Cast (RotaryEmbedDimHalf)", "to", int_type)
              .Add(
                  "CosCacheSliced = Slice(CosCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of cos
                                                                                                  // matrix:
                                                                                                  // [batch_size,
                                                                                                  // sequence_length,
                                                                                                  // rotary_embedding_dim
                                                                                                  // / 2]
              .Add(
                  "SinCacheSliced = Slice(SinCacheGather, Zero1D, RotaryEmbedDimHalfInt, Two1D)") // shape of sin
                                                                                                  // matrix:
                                                                                                  // [batch_size,
                                                                                                  // sequence_length,
                                                                                                  // rotary_embedding_dim
                                                                                                  // / 2]
              .Add("CosCacheUnsqueezed = Unsqueeze(CosCacheSliced, Two1D)") // shape of cos matrix: [batch_size,
                                                                            // sequence_length, 1, rotary_embedding_dim
                                                                            // / 2]
              .Add("SinCacheUnsqueezed = Unsqueeze(SinCacheSliced, Two1D)"); // shape of sin matrix: [batch_size,
                                                                             // sequence_length, 1, rotary_embedding_dim
                                                                             // / 2]

          // Create slices of inputs to multiply with sin and cos matrices based on interleaved parameter
          // Choose the correct slices based on interleaved parameter
          // real = cos_x * x1 - sin_x * x2
          // imag = sin_x * x1 + cos_x * x2
          if (interleaved == 0) {
            // For non-interleaved (basic) rotation, slices are created as follows,
            builder.Add(
                "X1, X2 = Split <axis = -1, num_outputs = 2> (XToRotate)"); // shape of X1 =
                                                                            // input[:,:,:,:rotary_embedding_dim/2],
                                                                            // X2 =
                                                                            // input[:,:,:,rotary_embedding_dim/2:rotary_embedding_dim]
          } else {
            // For interleaved rotation, slices are created as follows,
            builder.Const1D("One1D", (int64_t)1)
                .Const1D("AxesRotaryDim", (int64_t)3)
                .Add("RotaryEmbedDimInclusive = Add(RotaryEmbedDim, One1D)")
                .Add(
                    "X1 = Slice(XToRotate, Zero1D, RotaryEmbedDim, AxesRotaryDim, Two1D)") // shape of X1 =
                                                                                           // input[:,:,:,0:rotary_embedding_dim:2]
                .Add(
                    "X2 = Slice(XToRotate, One1D, RotaryEmbedDimInclusive, AxesRotaryDim, Two1D)"); // shape of
                                                                                                    // X2 =
                                                                                                    // input[:,:,:,1:rotary_embedding_dim:2]
          }

          builder.Add("CosX1 = Mul(CosCacheUnsqueezed, X1)")
              .Add("SinX2 = Mul(SinCacheUnsqueezed, X2)")
              .Add("Real = Sub(CosX1, SinX2)")
              .Add("SinX1 = Mul(SinCacheUnsqueezed, X1)")
              .Add("CosX2 = Mul(CosCacheUnsqueezed, X2)")
              .Add("Imaginary = Add(SinX1, CosX2)");

          // Insert the real and imaginary values into the original input to be rotated based on interleaved parameter
          if (interleaved == 0) {
            builder.Add("XRotated = Concat <axis = -1> (Real, Imaginary)");
          } else {
            builder
                .Add("RealInterleave = Unsqueeze(Real, NegOne)") // shape of indices =
                                                                 // input[:,:,:,0:rotary_embedding_dim:2, 1]
                .Add("ImaginaryInterleave = Unsqueeze(Imaginary, NegOne)") // shape of indices =
                                                                           // input[:,:,:,1:rotary_embedding_dim+1:2, 1]
                .Add("XRotatedInterleavedConcat = Concat <axis = -1> (RealInterleave, ImaginaryInterleave)")
                .Add("XRotatedShape = Shape(XToRotate)")
                .Add("XRotated = Reshape(XRotatedInterleavedConcat, XRotatedShape)");
          }

          // Combine rotated parts with non-rotated parts
          builder.Add("XConcat = Concat <axis = -1> (XRotated, XNoRotate)");

          if (is_input_4d == 0) {
            builder.Add("YTransposed = Identity(XConcat)");
          } else {
            builder.Add("YTransposed = Transpose <perm = [0, 2, 1, 3]> (XConcat)");
          }
          // Reshape back to 3D shape if input is a 3D tensor
          builder.Add("XShape = Shape(X)").Add("Y = Reshape(YTransposed, XShape)");

          schema.BuildFunction(functionProto);
          return true;
        }));

static const char* Attention_ver24_doc = R"DOC(

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.

With respect to KV cache update, this operator allows the following two use cases:

1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
key and value, respectively. Note that this only works correctly for the special case where the past key and value
do not contain padded tokens.
2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
as the maximum value of `nonpad_kv_seqlen`.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
 at_mask---Add              |
            |               |
  softcap (if provided)     |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Attention,
    24,
    OpSchema()
        .SetDoc(Attention_ver24_doc)
        .Attr(
            "is_causal",
            "If set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. "
            "The attention masking has the form of the upper left causal bias due to the alignment.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "scale",
            "Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent "
            "[numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "q_num_heads",
            "Number of heads of query. Must be used with 3D inputs of Q, K and V. ",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "kv_num_heads",
            "Number of heads of key and value. Must be used with 3D inputs of Q, K and V. ",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "softmax_precision",
            "The floating-point precision used in softmax computation. "
            "If softmax precision is not provided, the same precision as the input of softmax (Q and K) is used.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "softcap",
            "Softcap value for attention weights. Default value is 0.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Attr(
            "qk_matmul_output_mode",
            "If set to `0`, qk_matmul_output is the output of qk matmul. "
            "If set to `1`, qk_matmul_output includes the addition of the attention mask to the output of qk matmul. "
            "If set to `2`, qk_matmul_output is the output after the softcap operation. "
            "If set to `3`, qk_matmul_output is the output after the softmax operation. "
            "Default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "Q",
            "Query tensor. "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, q_hidden_size)`. "
            "For cases with a 3D input tensor, `q_hidden_size = q_num_heads * head_size`",
            "T1")
        .Input(
            1,
            "K",
            "Key tensor. "
            "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, k_hidden_size)`. "
            "For cases with a 3D input tensor, `k_hidden_size = kv_num_heads * head_size`",
            "T1")
        .Input(
            2,
            "V",
            "Value tensor. "
            "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, v_hidden_size)`. "
            "For cases with a 3D input tensor, `v_hidden_size = kv_num_heads * v_head_size`",
            "T2")
        .Input(
            3,
            "attn_mask",
            "Attention mask. "
            "Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length.` "
            "The last dimension can also be shorter than `total_sequence_length` and will be padded to `total_sequence_length` with negative infinity. "
            "Two types of masks are supported: a boolean mask where a value of `True` indicates that the element should take part in attention, "
            "or a float mask of the same type as query, key, value that is added to the attention score.",
            "U",
            OpSchema::Optional)
        .Input(
            4,
            "past_key",
            "past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`",
            "T1",
            OpSchema::Optional)
        .Input(
            5,
            "past_value",
            "past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`",
            "T2",
            OpSchema::Optional)
        .Input(
            6,
            "nonpad_kv_seqlen",
            "A vector of integers of shape `(batch_size,)` that indicates the number of valid (ie, non-padding) "
            "tokens in each sample. A padding mask can be derived from this. This should not be used together with "
            "`past_key` and `past_value` inputs or `present_key` and `present_value` outputs "
            "(See the KV cache use cases in the operator description).",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            0,
            "Y",
            "The output tensor . "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. "
            "For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`",
            "T1")
        .Output(
            1,
            "present_key",
            "Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T1",
            OpSchema::Optional)
        .Output(
            2,
            "present_value",
            "Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T2",
            OpSchema::Optional)
        .Output(
            3,
            "qk_matmul_output",
            "The output of QK matmul. "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T1",
            OpSchema::Optional)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q and K inputs types to float tensors.")
        .TypeConstraint("T2", OpSchema::all_float_types_ir4(), "Constrain V input types to float tensors.")
        .TypeConstraint(
            "U",
            OpSchema::all_non_complex_numeric_types_plus_bool_ir4(),
            "Constrain output 'mask' types to boolean tensors and input types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          int64_t kv_sequence_length = -1;
          ONNX_NAMESPACE::TensorShapeProto output_shape;
          ONNX_NAMESPACE::TensorShapeProto qk_matmul_shape;
          if (hasInputShape(ctx, 0)) {
            auto& query_shape = getInputShape(ctx, 0);
            auto& query_dims = query_shape.dim();
            if ((query_dims.size() != 3) && (query_dims.size() != 4)) {
              fail_shape_inference("Inputs 0 (query) shall be 3 or 4 dimensions");
            }

            if (query_dims.size() == 3) {
              auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
              if (q_num_heads_attr == nullptr) {
                fail_type_inference("3D inputs expected to have q_num_heads attribute.");
              }
              auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
              if (kv_num_heads_attr == nullptr) {
                fail_type_inference("3D inputs expected to have q_num_heads attribute.");
              }
            }

            *output_shape.add_dim() = query_dims[0]; // batch_size
            *output_shape.add_dim() = query_dims[1]; // num_heads for 4D, sequence_length for 3D

            *qk_matmul_shape.add_dim() = query_dims[0]; // batch_size

            if (hasInputShape(ctx, 1)) {
              auto& key_shape = getInputShape(ctx, 1);
              auto& key_dims = key_shape.dim();
              if ((key_dims.size() != 3) && (key_dims.size() != 4)) {
                fail_shape_inference("Inputs 1 (key) shall be 3 or 4 dimensions");
              }
            }

            if (hasInputShape(ctx, 2)) {
              auto& value_shape = getInputShape(ctx, 2);
              auto& value_dims = value_shape.dim();
              if ((value_dims.size() != 3) && (value_dims.size() != 4)) {
                fail_shape_inference("Inputs 2 (value) shall be 3 or 4 dimensions");
              }

              // Update Output Shape for 4D inputs
              // Input 0 (query) has shape (batch_size, q_num_heads, q_sequence_length, head_size)
              // Input 1 (key) has shape (batch_size, kv_num_heads, kv_sequence_length, head_size)
              // Input 2 (value) has shape (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
              // Output 0 has shape (batch_size, q_num_heads, q_sequence_length, v_head_size)
              if (value_dims.size() == 4 && query_dims.size() == 4) {
                kv_sequence_length = value_dims[2].dim_value();
                *output_shape.add_dim() = query_dims[2]; // sequence_length
                *output_shape.add_dim() = value_dims[3]; // head_size
                updateOutputShape(ctx, 0, output_shape);
                // Update qk_matmul_shape
                *qk_matmul_shape.add_dim() = query_dims[1]; // q_num_heads
                *qk_matmul_shape.add_dim() = query_dims[2]; // q_sequence_length
                qk_matmul_shape.add_dim()->set_dim_value(kv_sequence_length);
              }

              // Update Output Shape for 3D inputs
              // Input 0 (query) has shape (batch_size, q_sequence_length, q_hidden_size),
              // q_hidden_size = q_num_heads * head_size
              // Input 1 (key) has shape (batch_size, kv_sequence_length, k_hidden_size),
              // k_hidden_size = kv_num_heads * head_size
              // Input 2 (value) has shape (batch_size, kv_sequence_length, v_hidden_size),
              // v_hidden_size = kv_num_heads * v_head_size
              // Output 0 has shape (batch_size, q_sequence_length, hidden_size),
              // hidden_size = q_num_heads * v_head_size
              if (value_dims.size() == 3 && query_dims.size() == 3) {
                kv_sequence_length = value_dims[1].dim_value();
                auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
                if (q_num_heads_attr == nullptr) {
                  fail_type_inference("3D inputs expected to have q_num_heads attribute.");
                }
                auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
                if (kv_num_heads_attr == nullptr) {
                  fail_type_inference("3D inputs expected to have kv_num_heads attribute.");
                }
                int64_t q_num_heads = q_num_heads_attr->i();
                int64_t kv_num_heads = kv_num_heads_attr->i();
                // Calculate v_head_size
                int64_t v_head_size = value_dims[2].dim_value() / kv_num_heads;
                output_shape.add_dim()->set_dim_value(v_head_size * q_num_heads);
                updateOutputShape(ctx, 0, output_shape);
                // Update qk_matmul_shape
                qk_matmul_shape.add_dim()->set_dim_value(q_num_heads);
                *qk_matmul_shape.add_dim() = query_dims[1];
                qk_matmul_shape.add_dim()->set_dim_value(kv_sequence_length);
              }
            }
          }

          if (ctx.hasOutput(3)) { // has qk_matmul_output
            propagateElemTypeFromInputToOutput(ctx, 0, 3);
            updateOutputShape(ctx, 3, qk_matmul_shape);
          }

          if (ctx.hasOutput(1) && ctx.hasOutput(2)) { // has present outputs
            if (ctx.hasInput(4) && ctx.hasInput(5)) { // has past_key
              // copy the type from query to present key and value
              propagateElemTypeFromInputToOutput(ctx, 4, 1);
              propagateElemTypeFromInputToOutput(ctx, 5, 2);

              if (hasInputShape(ctx, 4) && hasInputShape(ctx, 5)) {
                auto& past_key_shape = getInputShape(ctx, 4);
                auto& past_key_dims = past_key_shape.dim();
                auto& past_value_shape = getInputShape(ctx, 5);
                auto& past_value_dims = past_value_shape.dim();

                // past key has shape (batch_size, kv_num_heads, past_sequence_length, head_size)
                if (past_key_dims.size() != 4) {
                  fail_shape_inference("The past_key input shall be 4 dimensions");
                }
                // past value has shape (batch_size, kv_num_heads, past_sequence_length, v_head_size)
                if (past_value_dims.size() != 4) {
                  fail_shape_inference("The past_value input shall be 4 dimensions");
                }

                if (kv_sequence_length > 0 && past_key_dims[2].has_dim_value()) {
                  int64_t total_sequence_length = kv_sequence_length + past_key_dims[2].dim_value();

                  ONNX_NAMESPACE::TensorShapeProto present_key_shape;
                  for (auto& dim : past_key_dims) {
                    *present_key_shape.add_dim() = dim;
                  }

                  ONNX_NAMESPACE::TensorShapeProto present_value_shape;
                  for (auto& dim : past_value_dims) {
                    *present_value_shape.add_dim() = dim;
                  }

                  if (ctx.hasOutput(3)) { // has qk_matmul_output with bias
                    qk_matmul_shape.mutable_dim(3)->set_dim_value(total_sequence_length);
                    updateOutputShape(ctx, 3, qk_matmul_shape);
                  }

                  // shape of present key/value is (batch_size, kv_num_heads, total_sequence_length, head_size)
                  present_key_shape.mutable_dim(2)->set_dim_value(total_sequence_length);
                  present_value_shape.mutable_dim(2)->set_dim_value(total_sequence_length);

                  updateOutputShape(ctx, 1, present_key_shape);
                  updateOutputShape(ctx, 2, present_value_shape);
                }
              }
            }
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads> (Q, K, V, attn_mask, past_key,
          // past_value) => (Y, present_key?, present_value?)
          int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          int64_t float_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

          // Get input types
          auto* t_qk = ctx.getInputType(0);
          if ((t_qk == nullptr) || (!t_qk->has_tensor_type()))
            return false;
          int64_t T1 = t_qk->tensor_type().elem_type();

          // Determine precision types for Softmax
          auto softmax_precision_attr = ctx.getAttribute("softmax_precision");
          int64_t softmax_precision = (softmax_precision_attr != nullptr) ? softmax_precision_attr->i() : T1;
          if ((softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
            return false; // Error

          auto mkbooltensor = [](bool val) -> ONNX_NAMESPACE::TensorProto {
            auto tp = ONNX_NAMESPACE::ToTensor(std::vector<bool>{val});
            tp.add_dims(1);
            return tp;
          };
          // If shape is 3D, q_num_heads and kv_num_heads is provided,
          // for 4D cases, set num_heads to zero for reshape purposes
          auto* q_num_heads_attr = ctx.getAttribute("q_num_heads");
          int64_t q_num_heads = (q_num_heads_attr != nullptr) ? q_num_heads_attr->i() : 0;
          auto* kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
          int64_t kv_num_heads = (kv_num_heads_attr != nullptr) ? kv_num_heads_attr->i() : 0;

          // Determine if input is 3D (requires reshape and transpose) or 4D (direct reshape)
          bool is_3d_input = (q_num_heads > 0 && kv_num_heads > 0);

          FunctionBuilder builder(functionProto);
          if (is_3d_input) {
            // For 3D inputs: First reshape to [batch_size, seq_length, num_heads, head_size]
            // then transpose to [batch_size, num_heads, seq_length, head_size]
            builder
                .Add("BatchSize = Shape <start = 0, end = 1> (Q)") // batch size
                .Const1D("QNumHeadsAttr", q_num_heads) // q_num_heads from attrs
                .Const1D("KVNumHeadsAttr", kv_num_heads) // kv_num_heads from attrs
                .Add("QSeqLen = Shape <start = -2, end = -1> (Q)") // q_sequence_length
                .Add("KVSeqLen = Shape <start = -2, end = -1> (K)") // kv_sequence_length
                .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

            builder.Add("QIntermediateShape = Concat <axis = 0> (BatchSize, QSeqLen, QNumHeadsAttr, NegOne)")
                .Add("KVIntermediateShape = Concat <axis = 0> (BatchSize, KVSeqLen, KVNumHeadsAttr, NegOne)")
                .Add("QIntermediate = Reshape (Q, QIntermediateShape)")
                .Add("KIntermediate = Reshape (K, KVIntermediateShape)")
                .Add("VIntermediate = Reshape (V, KVIntermediateShape)")
                // Then transpose to [batch_size, num_heads, seq_length, head_size]
                .Add("QReshaped = Transpose <perm = [0, 2, 1, 3]> (QIntermediate)")
                .Add("KReshaped = Transpose <perm = [0, 2, 1, 3]> (KIntermediate)")
                .Add("VReshaped = Transpose <perm = [0, 2, 1, 3]> (VIntermediate)");
          } else {
            // For 4D inputs: Already in desired shape [batch_size, num_heads, seq_length, head_size]
            builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");
            builder.Add("QSeqLen = Shape <start = -2, end = -1> (Q)");
          }

          builder
              .Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)") // q_num_heads
              .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)"); // kv_num_heads

          // Calculate scaling factor if scale attribute not provided
          auto scale_attr = ctx.getAttribute("scale");
          float scale = (scale_attr != nullptr) ? scale_attr->f() : static_cast<float>(1);
          builder
              .Add("QKHeadSize = Shape <start = 3, end = 4> (QReshaped)") // head_size for Q and K
              .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", float_type)
              .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
              .Const1D("One1D", static_cast<int64_t>(1))
              .Const1D("NegOne1D", static_cast<int64_t>(-1))
              .Const1D("One1DF", static_cast<float>(1))
              .Const1D("Zero1D", static_cast<int64_t>(0))
              .Add("CalculatedScale = Div(One1DF, SqrtHeadSize)")
              .Const("ScaleF", ToTensor<float>(scale))
              .Add(scale_attr != nullptr ? "ScaleFactor = Identity(ScaleF)" : "ScaleFactor = Identity(CalculatedScale)")
              .Add("ScaleFactorSqrt = Sqrt(ScaleFactor)")
              .Add("ScaleFactorF = Cast (ScaleFactorSqrt)", "to", T1);

          // Update key and value caches for past and present states

          if (ctx.hasInput(4)) {
            builder.Add("PresentKey = Concat <axis = 2> (past_key, KReshaped)");
          } else {
            builder.Add("PresentKey = Identity (KReshaped)");
          }
          if (ctx.hasOutput(1)) {
            builder.Add("present_key = Identity (PresentKey)");
          }

          if (ctx.hasInput(5)) {
            builder.Add("PresentValue = Concat <axis = 2> (past_value, VReshaped)");
          } else {
            builder.Add("PresentValue = Identity (VReshaped)");
          }
          if (ctx.hasOutput(2)) {
            builder.Add("present_value = Identity (PresentValue)");
          }

          builder.Add("NewKVSeqLen =  Shape <start = -2, end = -1> (PresentKey)");
          builder.Add("AttnBiasShape = Concat <axis = 0> (QSeqLen, NewKVSeqLen)");
          float neg_inf = -std::numeric_limits<float>::infinity();
          builder.Const1D("FloatNegInf", neg_inf);
          builder.Const1D("ScalarZero", 0.f);

          // If attn_mask is provided
          if (ctx.hasInput(3)) {
            auto* up = ctx.getInputType(3);
            if ((up == nullptr) || (!up->has_tensor_type()))
              return false;
            int64_t U = up->tensor_type().elem_type();
            builder.Add(
                U == ONNX_NAMESPACE::TensorProto_DataType_BOOL
                    ? "AttnBiasShort = Where(attn_mask, ScalarZero, FloatNegInf)"
                    : "AttnBiasShort = Identity(attn_mask)");
            // If attn_mask has a shorter kv sequence length, we pad it to NewKVSeqLen with FloatNegInf
            builder.Add("MaskKVSeqLen = Shape <start = -1> (attn_mask)")
                .Add("PaddingKVSeqLen = Sub(NewKVSeqLen, MaskKVSeqLen)")
                .Add("Pads = Concat <axis = 0> (Zero1D, PaddingKVSeqLen)")
                .Add("FloatNegInfCast = CastLike(FloatNegInf, AttnBiasShort)")
                .Add("AttnBias = Pad(AttnBiasShort, Pads, FloatNegInfCast, NegOne1D)");
          } else {
            builder.Add("AttnBias = ConstantOfShape(AttnBiasShape)");
          }

          // If is_causal set to true, the attention masking is a lower triangular matrix when the mask
          // is a square matrix. The attention masking has the form of the upper left causal bias due to
          // the alignment when the mask is a non-square matrix.
          // An error is thrown if both attn_mask and is_causal are set.
          auto* is_causal_attr = ctx.getAttribute("is_causal");
          int64_t is_causal = (is_causal_attr != nullptr) ? is_causal_attr->i() : 0;
          if (is_causal == 1) {
            builder.Add("BoolMask = ConstantOfShape(AttnBiasShape)", "value", mkbooltensor(1))
                .Add("BoolMaskTri = Trilu <upper = 0> (BoolMask, Zero1D)")
                .Add("MaskTri = Where(BoolMaskTri, ScalarZero, FloatNegInf)")
                .Add("AttnBiasCausal = Add(AttnBias, MaskTri)");
          } else {
            builder.Add("AttnBiasCausal = Identity(AttnBias)");
          }

          // Add padding mask if kv_nonpad_seqlen is provided
          if (ctx.hasInput(6)) {
            if (!is_3d_input) {
              builder.Add("KVSeqLen = Shape <start = -2, end = -1> (K)");
            }
            builder
                .Add("KVSeqLenExpanded = Unsqueeze(nonpad_kv_seqlen, One1D)") // [batch_size, 1]
                .Add("KVSeqLen0D = Squeeze(KVSeqLen)")
                .Const("Zero0D", static_cast<int64_t>(0))
                .Const("One0D", static_cast<int64_t>(1))
                .Add("Range = Range(Zero0D, KVSeqLen0D, One0D)") // [KVSeqLen,]
                .Add("PaddingMaskBool = Less(Range, KVSeqLenExpanded)") // [batch_size, KVSeqLen]
                .Add("PaddingMaskFloat = Where(PaddingMaskBool, ScalarZero, FloatNegInf)") // [batch_size, KVSeqLen]
                .Add("PaddingMask3D = Unsqueeze(PaddingMaskFloat, One1D)") // [batch_size, 1, KVSeqLen]
                .Add("PaddingMask4D = Unsqueeze(PaddingMask3D, One1D)") // [batch_size, 1, 1, KVSeqLen]
                .Add("AttnBiasCausalPad = Add(AttnBiasCausal, PaddingMask4D)");
          } else {
            builder.Add("AttnBiasCausalPad = Identity(AttnBiasCausal)");
          }
          builder.Add("AttnBiasT = Cast (AttnBiasCausalPad)", "to", T1);

          // Group Query Attention is applied if the following are satisfied
          // 1) q_num_heads != kv_num_heads
          // 2) q_num_heads % kv_num_heads == 0
          // 3) kv_num_heads == k_num_heads == v_num_heads
          builder.Add("NGQACond1 = Equal(QNumHeads, KVNumHeads)")
              .Add("GQACond1 = Not(NGQACond1)")
              .Add("DivNumHeads = Div(QNumHeads, KVNumHeads)")
              .Add("IDivNumHeads = Cast(DivNumHeads)", "to", int_type)
              .Add("RemainderNumHeads = Mod(QNumHeads, KVNumHeads)")
              .Add("GQACond2 = Equal(RemainderNumHeads, Zero1D)")
              .Add("GQACond = And(GQACond1, GQACond2)")
              .Add("InterleaveDim = Where(GQACond, IDivNumHeads, One1D)")
              .Add("InterleaveShape = Concat <axis = 0> (One1D, InterleaveDim, One1D, One1D)")
              .Add("KAttentionInput = Tile(PresentKey, InterleaveShape)")
              .Add("VAttentionInput = Tile(PresentValue, InterleaveShape)");

          // The following pattern is applied
          //      Q          K          V
          //      |          |          |
          //     Q*scale    K*scale     |
          //      |          |          |
          //      |       Transpose     |
          //      |          |          |
          //      ---MatMul---          |
          //            |               |
          // at_mask---Add              |
          //  softcap (if provided)     |
          //            |               |
          //            |               |
          //         Softmax            |
          //            |               |
          //            -----MatMul------
          //                    |
          //                    Y
          builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAttentionInput)")
              .Add("QScaled = Mul(QReshaped, ScaleFactorF)")
              .Add("KScaled = Mul(KTranspose, ScaleFactorF)")
              .Add("QKAttnWeight = MatMul(QScaled, KScaled)")
              .Add("QKAttnCast = Cast (QKAttnWeight)", "to", T1)
              .Add("QKAttnWeightWithBias = Add(QKAttnCast, AttnBiasT)");

          // Apply softcap if provided
          auto* softcap_attr = ctx.getAttribute("softcap");
          float softcap_val = (softcap_attr != nullptr) ? softcap_attr->f() : static_cast<float>(0);
          if (softcap_val != 0) {
            builder.Const1D("Softcap", softcap_val)
                .Add("SoftcapF = Cast (Softcap)", "to", T1)
                .Add("SoftcapDiv = Div(QKAttnWeightWithBias, SoftcapF)")
                .Add("SoftcapTanh = Tanh(SoftcapDiv)")
                .Add("QKAttnWeightSoftcap = Mul(SoftcapTanh, SoftcapF)");
          } else {
            builder.Add("QKAttnWeightSoftcap = Identity(QKAttnWeightWithBias)");
          }
          builder.Add("SoftmaxCast = Cast (QKAttnWeightSoftcap)", "to", softmax_precision)
              .Add("AttnWeightSoftmax = Softmax (SoftmaxCast)")
              .Add("SoftmaxOut = Cast (AttnWeightSoftmax)", "to", T1);

          // QK MatMul output if required
          auto* qk_matmul_output_mode_attr = ctx.getAttribute("qk_matmul_output_mode");
          int64_t qk_matmul_output_mode = (qk_matmul_output_mode_attr != nullptr) ? qk_matmul_output_mode_attr->i() : 0;
          if (ctx.hasOutput(3)) {
            if (qk_matmul_output_mode == 1) {
              builder.Add("qk_matmul_output = Identity(QKAttnWeightWithBias)");
            } else if (qk_matmul_output_mode == 2) {
              builder.Add("qk_matmul_output = Identity(QKAttnWeightSoftcap)");
            } else if (qk_matmul_output_mode == 3) {
              builder.Add("qk_matmul_output = Identity(AttnWeightSoftmax)");
            } else {
              builder.Add("qk_matmul_output = Identity(QKAttnWeight)");
            }
          }

          builder.Add("YPreReshape = MatMul(SoftmaxOut, VAttentionInput)");
          // Reshape Y to 3D if input is a 3D tensor
          if (is_3d_input) {
            builder.Add("YTranspose = Transpose <perm = [0, 2, 1, 3]> (YPreReshape)")
                .Add("YNewShape = Concat <axis = 0> (Zero1D, Zero1D, NegOne)")
                .Add("Y = Reshape(YTranspose, YNewShape)");
          } else {
            builder.Add("Y = Identity(YPreReshape)");
          }

          schema.BuildFunction(functionProto);
          return true;
        }));
} // namespace ONNX_NAMESPACE
