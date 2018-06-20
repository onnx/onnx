// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
static const char* Constant_ver1_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    1,
    OpSchema()
        .SetDoc(Constant_ver1_doc)
        .Attr(
            "value",
            "The value for the elements of the output tensor.",
            AttributeProto::TENSOR)
        .Output(
            0,
            "output",
            "Output tensor containing the same value of the provided tensor.",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto)
            return; // attribute not present
          if (!attr_proto->has_t())
            return; // attribute has no tensor value
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* RandomUniform_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniform,
    1,
    OpSchema()
        .SetDoc(RandomUniform_ver1_doc)
        .Attr(
            "low",
            "Lower boundary of the output values. If not specified, default is 0.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "high",
            "Upper boundary of the output values. If not specified, default is 1.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL)
        .Attr(
            "dtype",
            "The data type for the elements of the output tensor. If not specified, default is TensorProto::FLOAT.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::FLOAT))
        .Attr("shape", "The shape of the output tensor.", AttributeProto::INTS)
        .Output(
            0,
            "output",
            "Output tensor of random values drawn from uniform distribution",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        }));

static const char* RandomNormal_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormal,
    1,
    OpSchema()
        .SetDoc(RandomNormal_ver1_doc)
        .Attr(
            "mean",
            "The mean of the normal distribution. If not specified, default is 0.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "scale",
            "The standard deviation of the normal distribution. If not specified, default is 1.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL)
        .Attr(
            "dtype",
            "The data type for the elements of the output tensor. Default is TensorProto::FLOAT.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::FLOAT))
        .Attr("shape", "The shape of the output tensor.", AttributeProto::INTS)
        .Output(
            0,
            "output",
            "Output tensor of random values drawn from normal distribution",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          propagateShapeFromAttributeToOutput(ctx, "shape", 0);
        }));

static const char* RandomUniformLike_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a uniform distribution. 
The shape of the output tensor is copied from the shape of the input tensor, 
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided. 
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomUniformLike,
    1,
    OpSchema()
        .SetDoc(RandomUniformLike_ver1_doc)
        .Attr(
            "low",
            "Lower boundary of the output values. If not specified, default is 0.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "high",
            "Upper boundary of the output values. If not specified, default is 1.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use"
            "the data type of the input tensor.",
            AttributeProto::INT,
            OPTIONAL)
        .Input(
            0,
            "input",
            "Input tensor to copy shape and optionally type information from.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of random values drawn from uniform distribution",
            "T2")
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* RandomNormalLike_ver1_doc = R"DOC(
Generate a tensor with random values drawn from a normal distribution. 
The shape of the output tensor is copied from the shape of the input tensor, 
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided. 
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RandomNormalLike,
    1,
    OpSchema()
        .SetDoc(RandomNormalLike_ver1_doc)
        .Attr(
            "mean",
            "The mean of the normal distribution. If not specified, default is 0.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "scale",
            "The standard deviation of the normal distribution. If not specified, default is 1.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use"
            "the data type of the input tensor.",
            AttributeProto::INT,
            OPTIONAL)
        .Input(
            0,
            "input",
            "Input tensor to copy shape and optionally type information from.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of random values drawn from normal distribution",
            "T2")
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr)
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          else
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

static const char* Multinomial_ver7_doc = R"DOC(
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Multinomial,
    7,
    OpSchema()
        .SetDoc(Multinomial_ver7_doc)
        .Attr(
            "sample_size",
            "Number of times to sample.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "seed",
            "(Optional) Seed to the random generator, if not specified we will auto generate one.",
            AttributeProto::FLOAT,
            OPTIONAL)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor, if not specified, we will use int32.",
            AttributeProto::INT,
            static_cast<int64_t>(TensorProto::INT32))
        .Input(
            0,
            "input",
            "Input tensor with shape [batch_size, class_size], where class_size is the number of all possible outcomes. Each value along the axis zero represents the unnormalized log-probability of each corresponding outcome in a batch.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor with shape [batch_size, sample_size], where sample_size is the number of times to sample. Each value along the axis zero represents the outcome of the corresponding sample in a batch.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input types to float tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain output types to integral tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto dtype = ctx.getAttribute("dtype");
          auto dataType = TensorProto_DataType::TensorProto_DataType_INT32;
          if (dtype != nullptr) {
            dataType = static_cast<TensorProto_DataType>(dtype->i());
            if (dataType != TensorProto_DataType::TensorProto_DataType_INT32 &&
                dataType != TensorProto_DataType::TensorProto_DataType_INT64)
              fail_type_inference("Output type must be int32 or int64");
          }
          updateOutputElemType(ctx, 0, dataType);

          TensorShapeProto::Dimension batch_size, sample_size;
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 2)
              fail_shape_inference("Input tensor must have rank 2");
            batch_size = input_shape.dim(0);
          } // else statically-unknown batch-size
          sample_size.set_dim_value(getAttribute(ctx, "sample_size", 1));
          updateOutputShape(ctx, 0, {batch_size, sample_size});
        }));

} // namespace ONNX_NAMESPACE