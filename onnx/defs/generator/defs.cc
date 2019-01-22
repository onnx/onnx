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

static const char* ConstantOfShape_ver9_doc = R"DOC(
Generate a tensor with given value and shape.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConstantOfShape,
    9,
    OpSchema()
        .SetDoc(ConstantOfShape_ver9_doc)
        .Attr(
            "value",
            "(Optional) The value of the output elements."
            "Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32",
            AttributeProto::TENSOR,
            OPTIONAL)
        .Input(
            0,
            "input",
            "1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of shape specified by 'input'."
            "If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'."
            "If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype "
            "defaults to float32.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(int64)"},
            "Constrain input types.")
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
            "Constrain output types to be numerics.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getAttribute("value") != nullptr) {
            propagateElemTypeFromDtypeToOutput(
                ctx, ctx.getAttribute("value"), 0);
          } else {
            propagateElemTypeFromDtypeToOutput(ctx, TensorProto::FLOAT, 0);
          }

          // Shape inference based on input shape
          auto final_output_shape =
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const TensorProto* targetShapeInitializer = ctx.getInputData(0);
          if (!targetShapeInitializer) {
            // This is the case when exact shape input is not available.
            // In this case, if the number of dimensions can be infered 
            // from the input 'shape' tensor, then we add the same number
            // of dimensions (without any dim_value information) to the output.
            if (ctx.getInputType(0)->tensor_type().has_shape()) {
                auto& input_shape = getInputShape(ctx, 0);
                auto input_shape_dim_size = input_shape.dim_size();
                if(input_shape_dim_size > 1) {
                    fail_shape_inference("Shape input must be a one-dimensional tensor.");
                }
                if (input_shape.dim(0).has_dim_value()) {
                    const auto& input_shape_dim_value = input_shape.dim(0).dim_value();
                    if (input_shape_dim_value > 0) {
                        for (int i = 0; i < input_shape_dim_value; ++i) {
                            auto newdim = final_output_shape->add_dim();
                            (void)(newdim); // To eliminate "unused variable" compiler warning.
                        }
                    }
                }
            }
            return;
          }

          // This is the second case when exact shape data is available.
          // In this case, we extract the shape values from input tensor
          // and create output tensor of that shape.
          // First, extract target shape value.
          std::vector<int64_t> targetShape;
          if (targetShapeInitializer->has_raw_data()) {
            const std::string& bytes = targetShapeInitializer->raw_data();
            targetShape.insert(
                targetShape.end(),
                reinterpret_cast<const int64_t*>(bytes.c_str()),
                reinterpret_cast<const int64_t*>(bytes.c_str() + bytes.size()));
          } else {
            const auto& data = targetShapeInitializer->int64_data();
            targetShape.insert(targetShape.end(), data.begin(), data.end());
          }
          // Next, set output shape to the target shape.
          for (const int64_t& targetShapeElem : targetShape) {
            if(targetShapeElem > 0) {
              auto* new_dim = final_output_shape->add_dim();
              new_dim->set_dim_value(targetShapeElem);
            } else {
              // Check if value is less than -1; fail if so
              fail_shape_inference("Invalid shape value: ", targetShapeElem);
            }
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
