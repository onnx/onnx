// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
const char* pads_doc =
    "Padding for the beginning and ending along each axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.";
const char* auto_pad_doc =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
    "only intended to support legacy uses, and for framework authors, one is explicitly "
    "encouraged to use explicit padding specified in the pads attribute.";
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {

void convPoolTypeAndShapeInference(
    InferenceContext& ctx,
    bool use_dilation,
    bool require_kernel_shape) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (ctx.getNumOutputs() > 1) {
    // MaxPool with two outputs case.
    auto output_type = ctx.getOutputType(1);
    if (output_type->value_case() == TypeProto::kTensorType ||
        output_type->value_case() == TypeProto::VALUE_NOT_SET) {
      output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    }
  }

  // we need the first input shape for this inference.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  // if kernel shape is an input (and not attribute)
  // we need the shape of the second input.
  if (!require_kernel_shape && !hasNInputShapes(ctx, 2)) {
    return;
  }

  // don't bother with legacy auto_pad for now
  if (ctx.getAttribute("auto_pad")) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    fail_shape_inference("Input tensor must have atleast 2 dimensions");
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  // Pooling operations don't support dilation, only Conv. For
  // simplicity of the code, we just treat them as having all-1s
  // dilation.
  std::vector<int64_t> dilations;
  if (use_dilation && getRepeatedAttribute(ctx, "dilations", dilations)) {
    if (dilations.size() != n_input_dims) {
      fail_shape_inference("Attribute dilations has incorrect size");
    }
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      fail_shape_inference("Attribute pads has incorrect size");
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
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
    auto second_input_shape = ctx.getInputType(1)->tensor_type().shape();
    for (int i = 2; i < second_input_shape.dim_size(); ++i) {
      if (!second_input_shape.dim(i).has_dim_value()) {
        return;
      }
      kernel_shape.push_back(second_input_shape.dim(i).dim_value());
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
    auto& second_input_shape = getInputShape(ctx, 1);
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

    int64_t effective_kernel_size = kernel_shape[i];
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_size = (effective_kernel_size - 1) * dilations[i] + 1;

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions =
        (effective_input_size - effective_kernel_size) / strides[i];

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
    const char* additionalDescription) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
 {name} consumes an input tensor X and applies {opName} pooling across
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
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
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
        "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL);
    schema.Attr(
        "auto_pad",
        auto_pad_doc,
        AttributeProto::STRING,
        std::string("NOTSET"));
    schema.Attr("pads", pads_doc, AttributeProto::INTS, OPTIONAL);
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
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      convPoolTypeAndShapeInference(ctx, false, true);
    });
  };
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    1,
    OpSchema().FillUsing(PoolOpSchemaGenerator(
        "AveragePool",
        "average",
        "The output of each pooling window is divided by the number of elements exclude pad.")));

ONNX_OPERATOR_SET_SCHEMA(
    AveragePool,
    7,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "AveragePool",
            "average",
            "The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero)."))
        .Attr(
            "count_include_pad",
            "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.",
            AttributeProto::INT,
            static_cast<int64_t>(0)));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    1,
    OpSchema().FillUsing(PoolOpSchemaGenerator(
        "MaxPool",
        "max",
        "The output of each pooling window is maximum number of elements exclude pad.")));

ONNX_OPERATOR_SET_SCHEMA(
    MaxPool,
    8,
    OpSchema()
        .FillUsing(PoolOpSchemaGenerator(
            "MaxPool",
            "max",
            "The output of each pooling window is maximum number of elements exclude pad."))
        .Attr(
            "storage_order",
            "The storage order of the tensor. 0 is row major, and 1 is column major.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
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
    fail_type_inference(
            "MaxUnpool op must have either two or three inputs.");
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
        fail_type_inference(
              "'output_shape' must be rank 1 tensor.");
      }
      if (output_shape.dim((int)0).has_dim_value() && 
          static_cast<int>(output_shape.dim((int)0).dim_value()) != input_shape.dim_size()) {
          fail_shape_inference(
                  "'output_shape' must have same number of elements as the shape of input tensor X.");
      }
    }
    return; // 'output_shape' is specified as input. Actual shape will be determined at runtime.
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
    9,
    OpSchema()
        .SetDoc(MaxUnpool_ver9_doc)
        .Attr(
            "kernel_shape",
            "The size of the kernel along each axis.",
            AttributeProto::INTS)
        .Attr(
            "strides",
            "Stride along each axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "pads",
            pads_doc,
            AttributeProto::INTS,
            OPTIONAL)
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
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)"},
            "Constrain index tensor to int64")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          maxUnpoolShapeInference(ctx);
        }));
} // namespace ONNX_NAMESPACE

namespace ONNX_NAMESPACE {
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
        "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL);
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
      convPoolTypeAndShapeInference(ctx, false, true);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    2,
    OpSchema().FillUsing(LpPoolOpSchemaGenerator("LpPool")));

} // namespace ONNX_NAMESPACE

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
        "dilation value along each axis of the filter.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL);
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
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      convPoolTypeAndShapeInference(ctx, true, false);
    });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    Conv,
    1,
    OpSchema().FillUsing(ConvOpSchemaGenerator("a filter")));

} // namespace ONNX_NAMESPACE

void convTransposeShapeInference(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need at least two inputs to have a shape for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  // don't bother with legacy auto_pad for now
  if (ctx.getAttribute("auto_pad")) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();
  if (input_shape.dim_size() < 2) {
    return; // Input tensor should have at least two dimensions.
  }

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size() - 2);

  std::vector<int64_t> dilations;
  if (getRepeatedAttribute(ctx, "dilations", dilations)) {
    return; // we don't handle the dialations.
  }

  std::vector<int64_t> pads;
  if (getRepeatedAttribute(ctx, "pads", pads)) {
    if (pads.size() != n_input_dims * 2) {
      return;
    }
  } else {
    pads.assign(n_input_dims * 2, 0);
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

  std::vector<int64_t> output_shape;
  if (getRepeatedAttribute(ctx, "output_shape", output_shape)) {
    if (output_shape.size() != n_input_dims) {
      return;
    }
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
      ctx.getInputType(1)->tensor_type().shape().dim(
          1); // channels should be the second dim of second input.

  int size_of_output = static_cast<int>(output_shape.size());
  if (size_of_output > 0) {
    for (int i = 0; i < size_of_output; ++i) {
      if (output_shape[i] < input_shape.dim(i + 2).dim_value()) {
        // TODO: throw exception?
        return; // output shape value cannot be smaller than the input shape
                // value
      }

      final_output_shape->add_dim()->set_dim_value(output_shape[i]);
    }
    return; // assume no need to proceed further when the output shape is given.
  }

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    auto newdim = final_output_shape->add_dim();
    if (!input_shape.dim(2 + i).has_dim_value()) {
      continue;
    }

    int64_t newdim_value =
        strides[i] * (input_shape.dim(2 + i).dim_value() - 1);
    newdim_value += (output_padding[i] + kernel_shape[i]);
    newdim_value -= pads[i];
    newdim_value -= pads[i + kernel_shape_size];

    // add in the initial position
    newdim->set_dim_value(newdim_value);
  }
}

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator(
    const char* filter_desc) {
  return [=](OpSchema& schema) {
    std::string doc = R"DOC(
The convolution transpose operator consumes an input tensor and {filter_desc},
and computes the output. 

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
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
        "The zero-padding added to one side of the output."
        " This is also called adjs/adjustment in some frameworks.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "dilations",
        "dilation value along each axis of the filter.",
        AttributeProto::INTS,
        OPTIONAL);
    schema.Attr(
        "strides", "Stride along each axis.", AttributeProto::INTS, OPTIONAL);
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
    1,
    OpSchema().FillUsing(ConvTransposeOpSchemaGenerator("a filter")));

} // namespace ONNX_NAMESPACE

// For GlobalPool operations.
void gloablPoolTypeShapeInference(InferenceContext& ctx) {
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

namespace ONNX_NAMESPACE {
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
        "tensor. Dimensions will be N x C x 1 x 1",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { gloablPoolTypeShapeInference(ctx); });
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
        "tensor. Dimensions will be N x C x 1 x 1",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.SetDoc(doc);
    schema.TypeAndShapeInferenceFunction(
        [](InferenceContext& ctx) { gloablPoolTypeShapeInference(ctx); });
  };
}

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    2,
    OpSchema().FillUsing(
        GlobalLpPoolingOpSchemaGenerator("LpPool", "lp pool")));

static const char* BatchNormalization_ver7_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    7,
    OpSchema()
        .NumOutputs({1, 5})
        .SetDoc(BatchNormalization_ver7_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "spatial",
            "If true, compute the mean and variance across per activation. "
            "If false, compute the mean and variance across per feature over "
            "each mini-batch.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
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
            "If spatial is true, the dimension of scale is (C). "
            "If spatial is false, the dimensions of scale are "
            "(C x D1 x ... x Dn)",
            "T")
        .Input(
            2,
            "B",
            "If spatial is true, the dimension of bias is (C). "
            "If spatial is false, the dimensions of bias are "
            "(C x D1 x ... x Dn)",
            "T")
        .Input(
            3,
            "mean",
            "If spatial is true, the dimension of the running mean "
            "(training) or the estimated mean (testing) is (C). "
            "If spatial is false, the dimensions of the running mean "
            "(training) or the estimated mean (testing) are (C x D1 x ... x Dn).",
            "T")
        .Input(
            4,
            "var",
            "If spatial is true, the dimension of the running variance"
            "(training) or the estimated variance (testing) is (C). "
            "If spatial is false, the dimensions of the running variance"
            "(training) or the estimated variance (testing) are (C x D1 x ... x Dn).",
            "T")
        .Output(
            0,
            "Y",
            "The output tensor of the same shape as X",
            "T")
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

static const char* Dropout_ver7_doc = R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    7,
    OpSchema()
        .SetDoc(Dropout_ver7_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "ratio",
            "The ratio of random dropout",
            AttributeProto::FLOAT,
            0.5f)
        .Input(0, "data", "The input data as Tensor.", "T")
        .Output(0, "output", "The output.", "T")
        .Output(1, "mask", "The output mask.", "T", OpSchema::Optional)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

static const char* Flatten_ver9_doc = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    9,
    OpSchema()
        .SetDoc(Flatten_ver9_doc)
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
            "The value for axis must be in the range [0, R], where R is the rank of the input tensor. "
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

} // namespace ONNX_NAMESPACE
