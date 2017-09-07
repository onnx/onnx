// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;
using SupportType = onnx::OpSchema::SupportType;

OPERATOR_SCHEMA(ConstantFill)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
The operator fills the elements of the output tensor with a constant value
specified by the 'value' argument.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message. If the 'dtype' argument is not provided, the data type of
'value' is used.

The output tensor shape is specified by the 'shape' argument. If the number of
input is 1, the shape will be identical to that of the input at run time with
optional additional dimensions appended at the end as specified by 'extra_shape'
argument. In that case the 'shape' argument should not be set.

If input_as_shape is set to true, then the input should be a 1D tensor
containing the desired output shape (the dimensions specified in extra_shape
will also be appended)

NOTE: Currently, it supports data type of float, int32, int64, and bool.
)DOC")
    .Attr("value",
        "The value for the elements of the output tensor.",
        AttrType::FLOAT)
    .Attr(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.",
        AttrType::INT)
    .Attr(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.",
        AttrType::INTS)
    .Attr(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.",
        AttrType::INTS)
    .Attr(
        "input_as_shape",
        "1D tensor containing the desired output shape.  First input must be in "
        "CPU context.",
        AttrType::INT)
    .Input(0, "input", "Input tensor (optional) to provide shape information.")
    .Output(
            0,
            "output",
            "Output tensor of constant values specified by 'value'"
            "argument and its type is specified by the 'dtype' argument");

OPERATOR_SCHEMA(Constant)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(A constant tensor.)DOC")
    .Attr("value",
          "The value for the elements of the output tensor.",
          AttrType::TENSOR)
    .Output(
            0,
            "output",
            "Output tensor containing the same value of the provided tensor.");

OPERATOR_SCHEMA(Caffe2ConvTranspose)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
    The transposed convolution consumes an input vector, the filter blob, and
    the bias blob, and computes the output. Note that other parameters, such as
    the stride and kernel size, or the pads' sizes in each direction are not
    necessary for input because they are provided by the
    ConvTransposeUnpoolOpBase operator. Various dimension checks are done
    implicitly, and the sizes are specified in the Input docs for this operator.
    As is expected, the filter is deconvolved with a subset of the
    image and the bias is added; this is done throughout the image data and the
    output is computed. As a side note on the implementation layout:
    conv_transpose_op_impl.h is the templated implementation of the
    conv_transpose_op.h file, which is why they are separate files.
  )DOC")
    .Input(
        0,
        "X",
        "Input data blob from previous layer; has size "
        "(N x C x H x W), where N is the batch size, C is the number of channels, and"
        " H and W are the height and width. Note that this is for the NCHW usage. On "
        "the other hand, the NHWC Op has a different set of dimension constraints.")
    .Input(
        1,
        "filter",
        "The filter blob that will be used in the transposed "
        "convolution; has size (M x C x kH x kW), where C is the number of channels,"
        " and kH and kW are the height and width of the kernel.")
    .Input(
        2,
        "bias",
        "The 1D bias blob that is added through the convolution;"
        "has size (C)")
    .Output(
        0,
        "Y",
        "Output data blob that contains the result of the "
        "transposed convolution. The output dimensions are functions of the kernel"
        " size, stride size, and pad lengths.")
    .Attr("pads", "", AttrType::INTS)
    .Attr("kernel_shape", "", AttrType::INTS)
    .Attr("dilations", "", AttrType::INTS)
    .Attr("group", "", AttrType::INT)
    .Attr("strides", "", AttrType::INTS);

OPERATOR_SCHEMA(SpatialBN)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
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
        "The input 4-dimensional tensor of shape NCHW.")
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
            "The running mean after the spatial BN operator. Must be in-place "
            "with the input mean. Should not be used for testing.")
    .Output(2,
            "var",
            "The running variance after the spatial BN operator. Must be "
            "in-place with the input var. Should not be used for testing.")
    .Output(3,
            "saved_mean",
            "Saved mean used during training to speed up gradient "
            "computation. Should not be used for testing.")
    .Output(4,
            "saved_var",
            "Saved variance used during training to speed up "
            "gradient computation. Should not be used for testing.");

OPERATOR_SCHEMA(LRN)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1,2)
    .Attr("size", "", AttrType::INT)
    .Attr("alpha", "", AttrType::FLOAT)
    .Attr("beta", "", AttrType::FLOAT)
    .Attr("bias", "", AttrType::FLOAT);

OPERATOR_SCHEMA(GivenTensorFill)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .Input(0, "shape", "The shape of filled tensor")
    .Output(0, "X", "The filled tensor")
    .Attr("values", "", AttrType::FLOATS)
    .Attr("shape", "", AttrType::INTS)
    .Attr("input_as_shape", "", AttrType::INT)
    .Attr("extra_shape", "", AttrType::INTS)
    .AllowConsumed({{0, 0}});

OPERATOR_SCHEMA(FC)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully
connected layer with 2D weight matrix W and 1D bias vector b. That is,
the layer computes Y = X * W^T + b, where X has size (M x K),
W has size (N x K), b has size (N), and Y has size (M x N),
where M is often the batch size.
NOTE: X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}] where a_i \in N+ and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = M and a_1 * ... * a_{n-1} = K.
Lastly, even though b is a 1D vector of size N, it is copied/resized to
be size (M x N) implicitly and added to each vector in the batch.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
    .Attr(
        "axis",
        "(int32_t) default to 1; describes the axis of the inputs; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttrType::INT)
    .Attr(
        "axis_w",
        "(int32_t) default to 1; describes the axis of the weights; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size",
        AttrType::INT)
    .Input(
        0,
        "X",
        "input tensor that's coerced into a 2D matrix of size (MxK) "
        "as described above")
    .Input(
        1,
        "W",
        "2D blob of size (KxN) containing fully connected weight "
        "matrix")
    .Input(2, "b", "1D blob containing bias vector")
    .Output(0, "Y", "2D output tensor");

OPERATOR_SCHEMA(Normalize)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a matrix, apply L2-normalization along the last dimension.
)DOC");

OPERATOR_SCHEMA(Scale)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowConsumed({{0, 0}})
    .SetDoc(R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC")
    .Attr("scale",
          "(float, default 1.0) the scale to apply.",
          AttrType::FLOAT);

OPERATOR_SCHEMA(ChannelShuffle)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1)
    .NumOutputs(1)
    .Attr("kernel_shape",
          "The size of the kernel along each axis",
          AttrType::INTS)
    .Attr("group",
          "Number of channel groups",
          AttrType::INT);

OPERATOR_SCHEMA(RecurrentNetwork)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
Run the input network in a recurrent fashion. This can be used to
implement fairly general recurrent neural networks (RNNs).
The operator proceeds as follows.
- First, initialized the states from the input recurrent states
- For each timestep T, apply the links (that map offsets from input/output
tensors into the inputs/outputs for the `step` network)
- Finally, alias the recurrent states to the specified output blobs.
This is a fairly special-case meta-operator, and so the implementation
is somewhat complex. It trades of generality (and frankly usability)
against performance and control (compared to e.g. TF
dynamic_rnn, Theano scan, etc).
See the usage examples for a flavor of how to use it.
)DOC");

OPERATOR_SCHEMA(GRUUnit)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC")
    .Attr(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.",
        AttrType::INT)
    .Input(0, "hidden_prev", "The previous GRU hidden state.")
    .Input(
        1,
        "gates",
        "Unactivated gate outputs from forget, update, "
        "and output gates, pre-activation.")
    .Input(
        2,
        "seq_lengths",
        "Array of sequence lengths.  "
        "len(seq_lengths) should equal batch size N.")
    .Input(3, "t", "The timestep for this operation.")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.");

OPERATOR_SCHEMA(ATen)
    .SetSupportLevel(SupportType::EXPERIMENTAL)
    .AllowUncheckedAttributes()
    .SetDoc(R"DOC(
Experimental allowing ATen operations to be accessed directly from Caffe2
to allow for quick prototyping when ONNX is missing standard versions of
and op)DOC");
