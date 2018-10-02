// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
namespace ONNX_NAMESPACE {
static const char* Constant_ver9_doc = R"DOC(A constant tensor.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Constant,
    9,
    OpSchema()
        .SetDoc(Constant_ver9_doc)
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
            OpSchema::all_tensor_types(),
            "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto attr_proto = ctx.getAttribute("value");
          if (nullptr == attr_proto || !attr_proto->has_t())
            fail_shape_inference(
                "Attribute 'value' of Constant node must exist with 'Tensor' data.");
          const TensorProto& tensor_proto = attr_proto->t();
          updateOutputElemType(ctx, 0, tensor_proto.data_type());
          updateOutputShape(ctx, 0, tensor_proto);
        }));

static const char* ConstantLike_ver9_doc = R"DOC(
Generate a tensor with specific constant value. The value can be specified by the 'value' 
attribute. The shape of the output tensor is the same as the input tensor, if the input 
tensor is provided, or the shape provided in the 'shape' attribute (if both are provided, 
the input tensor shape takes precendence). The data type can be specified by the 'dtype' 
argument. If 'dtype' is not specified, then the type of input tensor is used. If input 
tensor is also not specified, then the type defaults to 'float'.

The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConstantLike,
    9,
    OpSchema()
        .SetDoc(ConstantLike_ver9_doc)
        .Attr(
            "value",
            "(Optional) The value for the elements of the output tensor.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor. If not specified,"
            "the data type of the input tensor T1 is used. If input tensor T1 is also not "
            "specified, then output tensor type defaults to 'float'.",
            AttributeProto::INT,
            OPTIONAL)
        .Attr(
            "shape",
            "(Optional) The shape of the output tensor. If input tensor T1 is provided, then"
            " 'shape' attribute is ignored and the output follows the shape of the input."
            " One of either input tensor T1 or 'shape' attribute must be provided.",
            AttributeProto::INTS,
            OPTIONAL)
        .Input(
            0,
            "input",
            "Input tensor to copy shape, and optionally, type information from."
            " One of either input tensor T1 or 'shape' attribute must be provided.",
            "T1",
            OpSchema::Optional)
        .Output(
            0,
            "output",
            "Output tensor, same shape as input tensor T1.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain input types. Strings and complex are not supported.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain output types. Strings and complex are not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr) {
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          } else {
            if (hasNInputShapes(ctx, 1)) {
              propagateElemTypeFromInputToOutput(ctx, 0, 0);
            }
          }
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

static const char* EyeLike_ver9_doc = R"DOC(
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D 
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the 
same as the input tensor. The data type can be specified by the 'dtype' argument. If 
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal 
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.

The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    EyeLike,
    9,
    OpSchema()
        .SetDoc(EyeLike_ver9_doc)
        .Attr(
            "k",
            "(Optional) Index of the diagonal to be populated with ones. Default is 0."
            " If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, "
            "k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "dtype",
            "(Optional) The data type for the elements of the output tensor. If not specified,"
            "the data type of the input tensor T1 is used. If input tensor T1 is also not"
            "specified, then type defaults to 'float'.",
            AttributeProto::INT,
            OPTIONAL)
        .Input(
            0,
            "input",
            "2D input tensor to copy shape, and optionally, type information from.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor, same shape as input tensor T1.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain input types. Strings and complex are not supported.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(bool)"},
            "Constrain output types. Strings and complex are not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("dtype") != nullptr) {
            propagateElemTypeFromAttributeToOutput(ctx, "dtype", 0);
          } else {
            propagateElemTypeFromInputToOutput(ctx, 0, 0);
          }
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() != 2) {
              fail_shape_inference("Input tensor must be 2-dimensional");
            }
          }
          propagateShapeFromInputToOutput(ctx, 0, 0);
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
            "Lower boundary of the output values.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "high",
            "Upper boundary of the output values.",
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
            "The mean of the normal distribution.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "scale",
            "The standard deviation of the normal distribution.",
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
            "Lower boundary of the output values.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "high",
            "Upper boundary of the output values.",
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
            "The mean of the normal distribution.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "scale",
            "The standard deviation of the normal distribution.",
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