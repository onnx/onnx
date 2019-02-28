// Copyright (c) Facebook Inc. and Microsoft Corporation.
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

	const char* pads_doc_v1 =
        "Padding for the beginning and ending along each axis, it can take any value greater "
        "than or equal to 0. The value represent the number of pixels added to the beginning "
        "and end part of the corresponding axis. `pads` format should be as follow "
        "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
        "added at the beginning of axis `i` and xi_end, the number of pixels added at "
        "the end of axis `i`. This attribute cannot be used simultaneously with "
        "auto_pad attribute. If not present, the padding defaults to 0 along start and end of each axis.";
    const char* auto_pad_doc_v1 =
        "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where "
        "default value is NOTSET, which means explicit padding is used. "
        "SAME_UPPER or SAME_LOWER mean pad the input so that the output size match the input."
        "In case of odd number add the extra padding at the end for SAME_UPPER and at the "
        "beginning for SAME_LOWER. VALID mean no padding. DEPRECATION NOTE: auto_pad is "
        "only intended to support legacy uses, and for framework authors, one is explicitly "
        "encouraged to use explicit padding specified in the pads attribute.";

	void convTransposeShapeInference_v1(InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // we need at least two inputs to have a shape for this inference.
      if (!hasNInputShapes(ctx, 2)) {
        return;
      }

      // don't bother with legacy auto_pad for now
      const auto* auto_pad_attr = ctx.getAttribute("auto_pad");
      if ((nullptr != auto_pad_attr) && (auto_pad_attr->s() != "NOTSET")) {
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
        for (auto i : dilations) {
          if (i != 1)
            return; // we don't handle dialations not 1.
        }
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
              return; // output shape value cannot be smaller than the input
                      // shape value
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
                output_padding[i] + kernel_shape[i] - pads[i] -
                pads[i + n_input_dims];
            final_output_shape->add_dim()->set_dim_value(output_shape_dim);
          } else {
            final_output_shape->add_dim();
          }
        }
        return;
      }
    }

	std::function<void(OpSchema&)> ConvTransposeOpSchemaGenerator_v1(
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
            "strides",
            "Stride along each axis.",
            AttributeProto::INTS,
            OPTIONAL);
        schema.Attr(
            "auto_pad",
            auto_pad_doc_v1,
            AttributeProto::STRING,
            std::string("NOTSET"));
        schema.Attr("pads", pads_doc_v1, AttributeProto::INTS, OPTIONAL);
        schema.Attr(
            "group",
            "number of groups input channels and output channels are divided into.",
            AttributeProto::INT,
            static_cast<int64_t>(1));
        schema.TypeAndShapeInferenceFunction(
            [](InferenceContext& ctx) { convTransposeShapeInference_v1(ctx); });
      };
    }

    ONNX_OPERATOR_SET_SCHEMA(
        ConvTranspose,
        1,
        OpSchema().FillUsing(ConvTransposeOpSchemaGenerator_v1("a filter")));

} // namespace ONNX_NAMESPACE
