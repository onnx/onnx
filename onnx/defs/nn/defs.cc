// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
const char* pads_doc =
    "Padding for the beginning and ending along each spatial axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.";
const char* auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding.";

void convPoolShapeInference(
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
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
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
    auto second_input_shape =
        ctx.getInputType(input2Idx)->tensor_type().shape();
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
    
  auto output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

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
    int64_t effective_input_size = input_shape.dim(2 + i).dim_value();
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = getAttribute(ctx, "ceil_mode", 0);

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions;

    if (ceil_mode == 1)
      strided_kernel_positions = (int64_t)(std::ceil(
          (effective_input_size - effective_kernel_shape[i]) / float(strides[i])));
    else
      strided_kernel_positions =
          (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    newdim->set_dim_value(1 + strided_kernel_positions);
  }

  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto second_output_shape =
        ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
    second_output_shape->CopyFrom(*output_shape);
  }
}

std::function<void(OpSchema&)> PoolOpSchemaGenerator(
    const char* name,
    const char* opName,
    const char* additionalDescription,
    bool use_dilation) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 {opName} pooling consisting of computing the {opName} on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
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
        use_dilation ? "((kernel_spatial_shape[i] - 1) * dilations[i] + 1)"
                     : "kernel_spatial_shape[i]");
    schema.SetDoc(doc);
    schema.Attr(
        "kernel_shape",
        "The size of the kernel along each axis.",
        AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc,
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "ceil_mode",
        "Wether to use ceil or floor (default) to compute the output shape.",
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
    11,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "AveragePool",
            "average",
            "The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).",
            false))
        .Attr(
            "count_include_pad",
            "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
            AttributeProto::INT,
            static_cast<int64_t>(0)));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    11,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "MaxPool",
            "max",
            "The output of each pooling window is maximum number of elements exclude pad.",
            true))
        .Attr(
            "storage_order",
            "The storage order of the tensor. 0 is row major, and 1 is column major.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "dilations",
            "Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
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
            OpSchema::Optional)
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "Constrain index tensor to int64"));

void maxUnpoolShapeInference(InferenceContext& ctx) {
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
    fail_shape_inference("Input tensor X must have atleast 2 dimensions.");
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
          static_cast<int>(output_shape.dim((int)0).dim_value()) !=
              input_shape.dim_size()) {
        fail_shape_inference(
            "'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be
            // determined at runtime.
  }

  auto final_output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(
          1); // channels should be the second dim of second input.

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value =
        strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += kernel_shape[i];
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

static const char* MaxUnpool_ver9_doc = R"DOC(
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MaxUnpool,
    11,
    OpSchema()
        .SetDoc(MaxUnpool_ver9_doc)
        .Attr(
            "kernel_shape",
            "The size of the kernel along each axis.",
            AttributeProto::INTS)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL)
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
            "T1")
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
            "T2")
        .Input(
            2,
            "output_shape",
            "The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, "
            "'pads' values are ignored.",
            "T2",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Output data tensor that contains the result of the unpooling.",
            "T1")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Constrain index tensor to int64")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          maxUnpoolShapeInference(ctx);
        }));

std::function<void(OpSchema&)> LpPoolOpSchemaGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 {name} consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";
    ReplaceAll(doc, "{name}", name);
    schema.SetDoc(doc);
    schema.Attr(
        "kernel_shape",
        "The size of the kernel along each axis.",
        AttributeProto::INTS);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc,
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "p",
        "p value of the Lp norm used to pool over the input data.",
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
        "dimensions are in the form of "
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
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      convPoolShapeInference(ctx, false, true, 0, 1);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    11,
    OpSchema().FillUsing(LpPoolOpSchemaGenerator("LpPool")));

// For ROI pool operations.
void roiPoolTypeShapeInference(InferenceContext& ctx) {
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
  auto output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *output_shape->add_dim() = rios_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);
  output_shape->add_dim()->set_dim_value(pooled_shape[0]);
  output_shape->add_dim()->set_dim_value(pooled_shape[1]);
}

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
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { roiPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    MaxRoiPool,
    1,
    OpSchema().FillUsing(RoiPoolOpSchemaGenerator("max")));

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
        "height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn). "
        "Optionally, if dimension denotation is "
        "in effect, the operation expects input data tensor "
        "to arrive with the dimension denotation of [DATA_BATCH, "
        "DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].",
        "T");
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
        "X.shape[1] == (W.shape[1] * group) == C "
        "(assuming zero based indices for the shape array). "
        "Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. ",
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
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc,
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr(
        "pads",
        pads_doc,
        AttributeProto::INTS,
        OPTIONAL);
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

ONNX_OPERATOR_SET_SCHEMA(
    Conv,
    11,
    OpSchema().FillUsing(ConvOpSchemaGenerator("a filter")));

static const char* QLinearConv_ver10_doc = R"DOC(
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
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
            "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M).",
            "T2")
        .Input(
            6,
            "y_scale",
            "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "tensor(float)")
        .Input(
            7,
            "y_zero_point",
            "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization.",
            "T3")
        .Input(
            8,
            "B",
            "Optional 1D bias to be added to the convolution, has size of M.",
            "T4",
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
            "Constrain input type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain filter type to 8-bit integer tensor.")
        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain output type to 8-bit integer tensor.")
        .TypeConstraint(
            "T4",
            {"tensor(int32)"},
            "Constrain bias type to 32-bit integer tensor.")
        .Attr(
            "auto_pad",
            auto_pad_doc,
            AttributeProto::STRING,
            std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext&
                                              ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(3);
          if (nullptr == x_type || nullptr == w_type ||
              x_type->value_case() != TypeProto::kTensorType ||
              w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference("inputs are expected to have tensor type.");
          }

          auto x_zero_point_type = ctx.getInputType(2);
          if (nullptr == x_zero_point_type ||
              x_zero_point_type->tensor_type().elem_type() !=
                  x_type->tensor_type().elem_type()) {
            fail_type_inference(
                "input and zero_point pair is expected to have be same type.");
          }

          auto w_zero_point_type = ctx.getInputType(5);
          if (nullptr == w_zero_point_type ||
              w_zero_point_type->tensor_type().elem_type() !=
                  w_type->tensor_type().elem_type()) {
            fail_type_inference(
                "weight and zero_point pair is expected to have same type.");
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
            "Scale tensor for input 'w'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, "
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
        .TypeConstraint(
            "T3",
            {"tensor(int32)"},
            "Constrain output y data type to 32-bit integer tensor.")
        .Attr(
            "auto_pad",
            auto_pad_doc,
            AttributeProto::STRING,
            std::string("NOTSET"))
        .Attr(
            "kernel_shape",
            "The shape of the convolution kernel. If not present, should be inferred from input 'w'.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "dilations",
            "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "strides",
            "Stride along each spatial axis. If not present, the stride defaults to 1 along each axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "pads",
            "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0."
            "The value represent the number of pixels added to the beginning and end part of the corresponding axis."
            "`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of"
            "pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`."
            "This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults"
            "to 0 along start and end of each spatial axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "group",
            "number of groups input channels and output channels are divided into. default is 1.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext&
                                              ctx) {
          auto x_type = ctx.getInputType(0);
          auto w_type = ctx.getInputType(1);
          auto y_type = ctx.getOutputType(0);
          if (nullptr == x_type || nullptr == w_type || nullptr == y_type ||
              x_type->value_case() != TypeProto::kTensorType ||
              w_type->value_case() != TypeProto::kTensorType) {
            fail_type_inference(
                "inputs are expected to have tensor type and output type should not be null.");
          }

          // Right now we only support int32
          y_type->mutable_tensor_type()->set_elem_type(
              TensorProto::INT32);

          convPoolShapeInference(ctx, true, false, 0, 1);
        }));

void convTransposeShapeInference(InferenceContext& ctx) {
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
    effective_kernel_shape[i] =
        (effective_kernel_shape[i] - 1) * dilations[i] + 1;
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
        int64_t total_pad = residual == 0
            ? effective_kernel_shape[i] - stride
            : effective_kernel_shape[i] - residual;
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

  auto final_output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  *final_output_shape->add_dim() = input_shape.dim(0);
  *final_output_shape->add_dim() =
      ctx.getInputType(1)->tensor_type().shape().dim(1) *
      group; // channels should be the second dim of second input multiply
             // group.

  int size_of_output;
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
        int64_t output_shape_dim =
            strides[i] * (input_shape.dim(i + 2).dim_value() - 1) +
            output_padding[i] + effective_kernel_shape[i] - pads[i] -
            pads[i + n_input_dims];
        final_output_shape->add_dim()->set_dim_value(output_shape_dim);
      } else {
        final_output_shape->add_dim();
      }
    }
    return;
  }
}

std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(
    const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    )DOC";
    ReplaceAll(doc, "{filter_desc}", filter_desc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from previous layer; has size (N x C x H x W)"
        ", where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the 2D image. "
        "Otherwise the size is (N x C x D1 x D2 ... x Dn)",
        "T");
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
        "Output data tensor that contains the result of the convolution. The "
        "output dimensions are functions of the kernel size, stride size, "
        "pad lengths and group count. "
        "The number of channels in the output should be equal to W.shape[1] * group "
        "(assuming zero based indices of the shape array)",
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
        "The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified "
        "pads values are ignored. See doc for details for equations to generate pads",
        AttributeProto::INTS,
        OPTIONAL);
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
        OPTIONAL);
    schema.Attr(
        "dilations",
        "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides",
        "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc,
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "group",
        "number of groups input channels and output channels are divided into.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { convTransposeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    ConvTranspose,
    11,
    OpSchema().FillUsing(ConvTransposeOpSchemaGenerator("a filter")));

// For GlobalPool operations.
void globalPoolTypeShapeInference(InferenceContext& ctx) {
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
  auto output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  *output_shape->add_dim() = input_shape.dim(0);
  *output_shape->add_dim() = input_shape.dim(1);

  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape->add_dim()->set_dim_value(1);
  }
}

std::function<void(OpSchema&)> GlobalPoolingOpSchemaGenerator(
    const char* op_type,
    const char* op) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
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
        "of the data. For non image case, the dimensions are "
        "in the form of (N x C x D1 x D2 ... Dn), "
        "where N is the batch size.",
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}
ONNX_OPERATOR_SET_SCHEMA(
    GlobalAveragePool,
    1,
    OpSchema().FillUsing(
        GlobalPoolingOpSchemaGenerator("AveragePool", "average")));
ONNX_OPERATOR_SET_SCHEMA(
    GlobalMaxPool,
    1,
    OpSchema().FillUsing(GlobalPoolingOpSchemaGenerator("MaxPool", "max")));

std::function<void(OpSchema&)> GlobalLpPoolingOpSchemaGenerator(
    const char* op_type,
    const char* op) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 Global{op_type} consumes an input tensor X and applies {op} pooling across
 the values in the same channel. This is equivalent to {op_type} with kernel size
 equal to the spatial dimension of input tensor.)DOC";
    ReplaceAll(doc, "{op_type}", op_type);
    ReplaceAll(doc, "{op}", op);
    schema.SetDoc(doc);
    schema.Attr(
        "p",
        "p value of the Lp norm used to pool over the input data.",
        AttributeProto::INT,
        static_cast<int64_t>(2));
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
        "T");
    schema.Output(
        0,
        "Y",
        "Output data tensor from pooling across the input "
        "tensor. The output tensor has the same rank as the input. "
        "The first two dimensions of output shape are the same as "
        "the input (N x C), while the other dimensions are all 1.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.SetDoc(doc);
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { globalPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    2,
    OpSchema().FillUsing(
        GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool")));

static const char* BatchNormalization_ver9_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    9,
    OpSchema()
        .NumOutputs({1, 5})
        .SetDoc(BatchNormalization_ver9_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT,
            1e-5f)
        .Attr(
            "momentum",
            "Factor used in computing the running mean and variance."
            "e.g., running_mean = running_mean * momentum + mean * (1 - momentum).",
            AttributeProto::FLOAT,
            0.9f)
        .Input(
            0,
            "X",
            "Input data tensor from the previous operator; "
            "dimensions are in the form of (N x C x D1 x D2 ... Dn), "
            "where N is the batch size, C is the number of channels. "
            "Statistics are computed for every channel of C over N and D1 to Dn dimensions. "
            "For image data, input dimensions become (N x C x H x W). "
            "The op also accepts single dimension input of size N in which case C is assumed to be 1",
            "T")
        .Input(1, "scale", "Scale tensor of shape (C).", "T")
        .Input(2, "B", "Bias tensor of shape (C).", "T")
        .Input(
            3,
            "mean",
            "running (training) or estimated (testing) mean tensor of shape (C).",
            "T")
        .Input(
            4,
            "var",
            "running (training) or estimated (testing) variance tensor of shape (C).",
            "T")
        .Output(0, "Y", "The output tensor of the same shape as X", "T")
        .Output(
            1,
            "mean",
            "The running mean after the BatchNormalization operator.",
            "T",
            OpSchema::Optional)
        .Output(
            2,
            "var",
            "The running variance after the BatchNormalization operator.",
            "T",
            OpSchema::Optional)
        .Output(
            3,
            "saved_mean",
            "Saved mean used during training to speed up gradient "
            "computation.",
            "T",
            OpSchema::Optional)
        .Output(
            4,
            "saved_var",
            "Saved variance used during training to speed up "
            "gradient computation.",
            "T",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          // TODO in training mode, it may be possible to infer some of
          // the other outputs as well.
        }));

static const char* InstanceNormalization_ver6_doc = R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    6,
    OpSchema()
        .SetDoc(InstanceNormalization_ver6_doc)
        .Attr(
            "epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttributeProto::FLOAT,
            1e-5f)
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
            "T")
        .Input(
            1,
            "scale",
            "The input 1-dimensional scale tensor of size C.",
            "T")
        .Input(2, "B", "The input 1-dimensional bias tensor of size C.", "T")
        .Output(
            0,
            "output",
            "The output tensor of the same shape as input.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
        }));

static const char* LpNormalization_ver1_doc = R"DOC(
Given a matrix, apply Lp-normalization along the provided axis.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LpNormalization,
    1,
    OpSchema()
        .Input(0, "input", "Input matrix", "T")
        .Output(0, "output", "Matrix after normalization", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .SetDoc(LpNormalization_ver1_doc)
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
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
        }));

static const char* Dropout_ver10_doc = R"DOC(
Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    10,
    OpSchema()
        .SetDoc(Dropout_ver10_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "ratio",
            "The ratio of random dropout",
            AttributeProto::FLOAT,
            0.5f)
        .Input(0, "data", "The input data as Tensor.", "T")
        .Output(0, "output", "The output.", "T")
        .Output(1, "mask", "The output mask.", "T1", OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T1",
            {"tensor(bool)"},
            "Constrain output mask types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
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
        .Attr(
            "lambd",
            "The lambd value for the Shrink formulation. Default is 0.5.",
            AttributeProto::FLOAT,
            0.5f)
        .Attr(
            "bias",
            "The bias value added to output. Default is 0.",
            AttributeProto::FLOAT,
            0.0f)
        .Input(0, "input", "The input data as Tensor.", "T")
        .Output(0, "output", "The output.", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_numeric_types(),
            "Constrains input to only numeric types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Flatten_ver11_doc = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    11,
    OpSchema()
        .SetDoc(Flatten_ver11_doc)
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
            OpSchema::all_tensor_types(),
            "Constrain input and output to all tensor types.")
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
            fail_shape_inference(
                "Invalid value(", axis, ") for attribute 'axis'");
          }
          // TODO: is the operation defined for input-rank < 2?
          updateOutputShape(
              ctx,
              0,
              {multiplyDims(input_shape, 0, axis),
               multiplyDims(input_shape, axis, rank)});
        }));

static const char* LRN_ver1_doc = R"DOC(
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LRN,
    1,
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
            "T")
        .Output(
            0,
            "Y",
            "Output tensor, which has the shape and type as input tensor",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output "
            " types to float tensors.")
        .SetDoc(LRN_ver1_doc)
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
        .Input(0, "X", "Input for n-gram extraction", "T")
        .Output(0, "Y", "Ngram results", "T1")
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
            OPTIONAL)
        .Attr(
            "pool_int64s",
            "List of int64 n-grams learned from the training set. Either this or pool_strings attributes must be present but not both. "
            "It's an 1-D tensor starting with the collections of all 1-grams and ending with the collections of n-grams. "
            "The i-th element in pool stores the n-gram that should be mapped to coordinate ngram_indexes[i] in the output vector.",
            AttributeProto::INTS,
            OPTIONAL)
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
            OPTIONAL)
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
                !std::all_of(
                    ngram_indexes.cbegin(),
                    ngram_indexes.cend(),
                    [](int64_t i) { return i >= 0; })) {
              fail_shape_inference(
                  "ngram_indexes must be non-empty with no negative values");
            }

            auto greatest_hit =
                std::max_element(ngram_indexes.cbegin(), ngram_indexes.cend());
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

static const char* StringNormalizer_ver10_doc = R"DOC(
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    StringNormalizer,
    10,
    OpSchema()
        .Input(0, "X", "UTF-8 strings to normalize", "tensor(string)")
        .Output(0, "Y", "UTF-8 Normalized strings", "tensor(string)")
        .Attr(
            std::string("case_change_action"),
            std::string(
                "string enum that cases output to be lowercased/uppercases/unchanged."
                " Valid values are \"LOWER\", \"UPPER\", \"NONE\". Default is \"NONE\""),
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            std::string("is_case_sensitive"),
            std::string(
                "Boolean. Whether the identification of stop words in X is case-sensitive. Default is false"),
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "stopwords",
            "List of stop words. If not set, no word would be removed from X.",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "locale",
            "Environment dependent string that denotes the locale according to which output strings needs to be upper/lowercased."
            "Default en_US or platform specific equivalent as decided by the implementation.",
            AttributeProto::STRING,
            OPTIONAL)
        .SetDoc(StringNormalizer_ver10_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(TensorProto::STRING);
          if (!hasInputShape(ctx, 0)) {
            return;
          }
          TensorShapeProto output_shape;
          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto dim_size = input_shape.dim_size();
          // Last axis dimension is unknown if we have stop-words since we do
          // not know how many stop-words are dropped
          if (dim_size == 1) {
            // Unknown output dimension
            output_shape.add_dim();
          } else if (dim_size == 2) {
            // Copy B-dim
            auto& b_dim = input_shape.dim(0);
            if (!b_dim.has_dim_value() || b_dim.dim_value() != 1) {
              fail_shape_inference(
                  "Input shape must have either [C] or [1,C] dimensions where C > 0");
            }
            *output_shape.add_dim() = b_dim;
            output_shape.add_dim();
          } else {
            fail_shape_inference(
                "Input shape must have either [C] or [1,C] dimensions where C > 0");
          }
          updateOutputShape(ctx, 0, output_shape);
        }));

static const char* mvn_ver9_doc = R"DOC(
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
)DOC";

static std::vector<int64_t> mvn_default_axes = {0, 2, 3};

ONNX_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization,
    9,
    OpSchema()
        .SetDoc(mvn_ver9_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Attr(
            "axes",
            "A list of integers, along which to reduce. The default is to "
            "caculate along axes [0,2,3] for calculating mean and variance "
            "along each channel. Two variables with the same C-coordinate "
            "are associated with the same mean and variance.",
            AttributeProto::INTS,
            mvn_default_axes)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to all numeric tensors.")
        .FunctionBody(FunctionBodyHelper::BuildNodes(
            {// nodes: {outputs, op, inputs, attributes}
             FunctionBodyHelper::Const<float>("Exponent", 2.0f),
             FunctionBodyHelper::Const<float>("Epsilon", float(1e-9)),
             {{"X_RM"},
              "ReduceMean",
              {"X"},
              {MakeRefAttribute("axes", AttributeProto::INTS)}},
             {{"EX_squared"}, "Pow", {"X_RM", "Exponent"}},
             {{"X_squared"}, "Pow", {"X", "Exponent"}},
             {{"E_Xsquared"},
              "ReduceMean",
              {"X_squared"},
              {MakeRefAttribute("axes", AttributeProto::INTS)}},
             {{"Variance"}, "Sub", {"E_Xsquared", "EX_squared"}},
             {{"STD"}, "Sqrt", {"Variance"}},
             {{"X_variance"}, "Sub", {"X", "X_RM"}},
             {{"Processed_STD"}, "Add", {"STD", "Epsilon"}},
             {{"Y"}, "Div", {"X_variance", "Processed_STD"}}})));

} // namespace ONNX_NAMESPACE
