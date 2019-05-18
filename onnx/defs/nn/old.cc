// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
const char* pads_doc_old =
    "Padding for the beginning and ending along each axis, it can take any value greater "
    "than or equal to 0. The value represent the number of pixels added to the beginning "
    "and end part of the corresponding axis. `pads` format should be as follow "
    "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
    "added at the beginning of axis `i` and xi_end, the number of pixels added at "
    "the end of axis `i`. This attribute cannot be used simultaneously with "
    "auto_pad attribute.";
const char* auto_pad_doc_old =
    "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
    "default value is NOTSET, which means explicit padding is used. "
    "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
    "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
    "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
    "only intended to support legacy uses, and for framework authors, one is explicitly "
    "encouraged to use explicit padding specified in the pads attribute.";

static const char* LpPool_ver1_doc = R"DOC(
 LpPool consumes an input tensor X and applies Lp pooling across the
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LpPool,
    1,
    OpSchema()
        .SetDoc(LpPool_ver1_doc)
        .Attr(
            "kernel_shape",
            "The size of the kernel along each axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "strides",
            "Stride along each axis.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "auto_pad",
            auto_pad_doc_old,
            AttributeProto::STRING,
            std::string("NOTSET"))
        .Attr("pads", pads_doc_old, AttributeProto::INTS, OPTIONAL)
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
            "Constrain input and output types to float tensors."));

static const char* GlobalLpPool_ver1_doc = R"DOC(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GlobalLpPool,
    1,
    OpSchema()
        .SetDoc(GlobalLpPool_ver1_doc)
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
            "Constrain input and output types to float tensors."));

static const char* BatchNormalization_ver1_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    1,
    OpSchema()
        .NumOutputs({1, 5})
        .SetDoc(BatchNormalization_ver1_doc)
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
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
            "Constrain input and output types to float tensors."));

static const char* InstanceNormalization_ver1_doc = R"DOC(
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    InstanceNormalization,
    1,
    OpSchema()
        .SetDoc(InstanceNormalization_ver1_doc)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
        .Input(
            1,
            "scale",
            "The input 1-dimensional scale tensor of size C.",
            "T")
        .Input(2, "B", "The input 1-dimensional bias tensor of size C.", "T")
        .Output(
            0,
            "output",
            "The output 4-dimensional tensor of the same shape as input.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors."));

static const char* Dropout_old_doc = R"DOC(
Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    1,
    OpSchema()
        .SetDoc(Dropout_old_doc)
        .Attr(
            "ratio",
            "(float, default 0.5) the ratio of random dropout",
            AttributeProto::FLOAT,
            0.5f)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
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
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Dropout,
    6,
    OpSchema()
        .SetDoc(Dropout_old_doc)
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
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

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

static const char* BatchNorm_ver6_doc = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    BatchNormalization,
    6,
    OpSchema()
        .NumOutputs({1, 5})
        .SetDoc(BatchNorm_ver6_doc)
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
        .Output(0, "Y", "The output tensor of the same shape as X.", "T")
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
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateShapeAndTypeFromFirstInput(ctx);
          // TODO in training mode, it may be possible to infer some of
          // the other outputs as well.
        }));

static const char* Flatten_ver1_doc = R"DOC(
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Flatten,
    1,
    OpSchema()
        .SetDoc(Flatten_ver1_doc)
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


void maxUnpoolShapeInference_old(InferenceContext& ctx) {
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
        .Attr("pads", pads_doc_old, AttributeProto::INTS, OPTIONAL)
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
          maxUnpoolShapeInference_old(ctx);
        }));
} // namespace ONNX_NAMESPACE
