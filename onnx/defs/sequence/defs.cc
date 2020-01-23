// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include <numeric>
#include <algorithm>

namespace ONNX_NAMESPACE {

static const char* SequenceEmpty_ver11_doc = R"DOC(
Construct an empty tensor sequence, with given data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceEmpty,
    11,
    OpSchema()
        .SetDoc(SequenceEmpty_ver11_doc)
        .Attr(
            "dtype",
            "(Optional) The data type of the tensors in the output sequence. "
            "The default type is 'float'.",
            AttributeProto::INT,
            OPTIONAL)
        .Output(
            0,
            "output",
            "Empty sequence.",
            "S")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto* attr_proto = ctx.getAttribute("dtype");
          auto elem_type = TensorProto::FLOAT;
          if (nullptr != attr_proto) {
            if (!attr_proto->has_i()) {
              fail_type_inference(
                  "Attribute dtype should be of integer type and specify a type.");
            }
            auto attr_value = attr_proto->i();
            elem_type = static_cast<TensorProto_DataType>(attr_value);
          }
          ctx.getOutputType(0)
              ->mutable_sequence_type()
              ->mutable_elem_type()
              ->mutable_tensor_type()
              ->set_elem_type(elem_type);
        }));

static const char* SequenceConstruct_ver11_doc = R"DOC(
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceConstruct,
    11,
    OpSchema()
        .SetDoc(SequenceConstruct_ver11_doc)
        .Input(
            0,
            "inputs",
            "Tensors.",
            "T",
            OpSchema::Variadic)
        .Output(
            0,
            "output_sequence",
            "Sequence enclosing the input tensors.",
            "S")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input types to any tensor type.")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs < 1) {
            fail_type_inference("SequenceConstruct is expected to have at least 1 input.");
          }

          std::vector<int> input_elem_types;
          for (size_t i = 0; i < numInputs; ++i) {
            input_elem_types.emplace_back(ctx.getInputType(i)->tensor_type().elem_type());
          }
          if (std::adjacent_find(input_elem_types.begin(), input_elem_types.end(), std::not_equal_to<int>()) != input_elem_types.end()) {
            // not all input elem types are the same.
            fail_type_inference("Element type of inputs are expected to be the same.");
          }

          auto* output_tensor_type =
              ctx.getOutputType(0)
                  ->mutable_sequence_type()
                  ->mutable_elem_type()
                  ->mutable_tensor_type();

          output_tensor_type->set_elem_type(static_cast<TensorProto_DataType>(input_elem_types[0]));

          if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }

          *(output_tensor_type->mutable_shape()) = ctx.getInputType(0)->tensor_type().shape();

          for (size_t i = 1; i < numInputs; ++i) {
            const auto& input_shape = ctx.getInputType(i)->tensor_type().shape();
            UnionShapeInfo(input_shape, *output_tensor_type);
          }
        }));

static const char* SequenceInsert_ver11_doc = R"DOC(
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceInsert,
    11,
    OpSchema()
        .SetDoc(SequenceInsert_ver11_doc)
        .Input(
            0,
            "input_sequence",
            "Input sequence.",
            "S")
        .Input(
            1,
            "tensor",
            "Input tensor to be inserted into the input sequence.",
            "T")
        .Input(
            2,
            "position",
            "Position in the sequence where the new tensor is inserted. "
            "It is optional and default is to insert to the back of the sequence. "
            "Negative value means counting positions from the back. "
            "Accepted range in `[-n, n]`, "
            "where `n` is the number of tensors in 'input_sequence'. "
            "It is an error if any of the index values are out of bounds. "
            "It must be a scalar(tensor of empty shape).",
            "I",
            OpSchema::Optional)
        .Output(
            0,
            "output_sequence",
            "Output sequence that contains the inserted tensor at given position.",
            "S")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "I",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain position to integral tensor. It must be a scalar(tensor of empty shape).")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto seq_elem_type = 
              ctx.getInputType(0)->sequence_type().elem_type().tensor_type().elem_type();
          const auto tensor_elem_type =
              ctx.getInputType(1)->tensor_type().elem_type();
          if (seq_elem_type != tensor_elem_type) {
            fail_type_inference(
                "Input Sequence and Tensor are expected to have the same elem type. Sequence=",
                seq_elem_type,
                " Tensor=",
                tensor_elem_type);
          }

          auto *output_tensor_type =
              ctx.getOutputType(0)
                  ->mutable_sequence_type()
                  ->mutable_elem_type()
                  ->mutable_tensor_type();
          output_tensor_type->set_elem_type(seq_elem_type);

          if (!hasInputShape(ctx, 0) || !hasInputShape(ctx, 1)) {
            return;
          }

          *(output_tensor_type->mutable_shape()) = 
              ctx.getInputType(0)->sequence_type().elem_type().tensor_type().shape();

          UnionShapeInfo(ctx.getInputType(1)->tensor_type().shape(), *output_tensor_type);
        }));

static const char* SequenceAt_ver11_doc = R"DOC(
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceAt,
    11,
    OpSchema()
        .SetDoc(SequenceAt_ver11_doc)
        .Input(
            0,
            "input_sequence",
            "Input sequence.",
            "S")
        .Input(
            1,
            "position",
            "Position of the tensor in the sequence. "
            "Negative value means counting positions from the back. "
            "Accepted range in `[-n, n - 1]`, "
            "where `n` is the number of tensors in 'input_sequence'. "
            "It is an error if any of the index values are out of bounds. "
            "It must be a scalar(tensor of empty shape).",
            "I")
        .Output(
            0,
            "tensor",
            "Output tensor at the specified position in the input sequence.",
            "T")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "I",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain position to integral tensor. It must be a scalar(tensor of empty shape).")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->CopyFrom(ctx.getInputType(0)->sequence_type().elem_type());
        }));

static const char* SequenceErase_ver11_doc = R"DOC(
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceErase,
    11,
    OpSchema()
        .SetDoc(SequenceErase_ver11_doc)
        .Input(
            0,
            "input_sequence",
            "Input sequence.",
            "S")
        .Input(
            1,
            "position",
            "Position of the tensor in the sequence. "
            "Negative value means counting positions from the back. "
            "Accepted range in `[-n, n - 1]`, "
            "where `n` is the number of tensors in 'input_sequence'. "
            "It is an error if any of the index values are out of bounds. "
            "It must be a scalar(tensor of empty shape).",
            "I",
            OpSchema::Optional)
        .Output(
            0,
            "output_sequence",
            "Output sequence that has the tensor at the specified position removed.",
            "S")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "I",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain position to integral tensor. It must be a scalar(tensor of empty shape).")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->CopyFrom(*ctx.getInputType(0));
        }));

static const char* SequenceLength_ver11_doc = R"DOC(
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SequenceLength,
    11,
    OpSchema()
        .SetDoc(SequenceLength_ver11_doc)
        .Input(
            0,
            "input_sequence",
            "Input sequence.",
            "S")
        .Output(
            0,
            "length",
            "Length of input sequence. It must be a scalar(tensor of empty shape).",
            "I")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain to any tensor type.")
        .TypeConstraint(
            "I",
            {"tensor(int64)"},
            "Constrain output to integral tensor. It must be a scalar(tensor of empty shape).")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::INT64);
          output_tensor_type->mutable_shape()->Clear();
        }));

// Updated operators that consume/produce sequence of tensors.

static const char* SplitToSequence_ver11_doc =
    R"DOC(Split a tensor into a sequence of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into equally sized chunks(if possible).
Last chunk will be smaller if the 'input' size along the given axis 'axis' is not divisible
by 'split'.
Otherwise, the tensor is split into 'size(split)' chunks, with lengths of the parts on 'axis'
specified in 'split'. In this scenario, the sum of entries in 'split' must be equal to the
dimension size of input tensor on 'axis'.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    SplitToSequence,
    11,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T")
        .Input(
            1,
            "split",
            "Length of each output. "
            "It can be either a scalar(tensor of empty shape), or a 1-D tensor. "
            "All values must be positive. ",
            "I",
            OpSchema::Optional)
        .Output(
            0,
            "output_sequence",
            "One or more outputs forming a sequence of tensors after splitting",
            "S")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain input types to all tensor types.")
        .TypeConstraint(
            "I",
            {"tensor(int32)", "tensor(int64)"},
            "Constrain split size to integral tensor.")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain output types to all tensor types.")
        .Attr(
            "axis",
            "Which axis to split on. "
            "A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1].",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "keepdims",
            "Keep the split dimension or not. Default 1, which means we keep split dimension. "
            "If input 'split' is specified, this attribute is ignored.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .SetDoc(SplitToSequence_ver11_doc)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          ctx.getOutputType(0)
              ->mutable_sequence_type()
              ->mutable_elem_type()
              ->mutable_tensor_type()
              ->set_elem_type(elem_type);

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          const auto& inputShape = ctx.getInputType(0)->tensor_type().shape();

          int r = inputShape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis > r - 1) {
            fail_type_inference(
                "Invalid value of attribute 'axis'. Rank=",
                r,
                " Value=",
                axis);
          }
          if (axis < 0) {
            axis += r;
          }

          size_t num_inputs = ctx.getNumInputs();
          int64_t splitSize = 1;
          int64_t keepdims = 1;
          if (num_inputs == 1) {
            // input split is omitted, default to split by 1.
            auto attr_proto = ctx.getAttribute("keepdims");
            if (attr_proto) {
              keepdims = attr_proto->i();
            }
          } else {
            splitSize = [&]() -> int64_t {
              // Need input split shape info and initializer data to infer split sizes.
              if (!ctx.getInputType(1)->tensor_type().has_shape()) {
                return -1;
              }
              const TensorProto* splitInitializer = ctx.getInputData(1);
              if (nullptr == splitInitializer || !splitInitializer->has_data_type()) {
                return -1;
              }

              std::vector<int64_t> splitSizes;
              if (splitInitializer->data_type() == TensorProto::INT64) {
                const auto& data = ParseData<int64_t>(splitInitializer);
                splitSizes.insert(splitSizes.end(), data.begin(), data.end());
              } else if (splitInitializer->data_type() == TensorProto::INT32) {
                const auto& data = ParseData<int32_t>(splitInitializer);
                splitSizes.insert(splitSizes.end(), data.begin(), data.end());
              } else {
                // unaccepted data type
                fail_shape_inference(
                    "Only supports `int32_t` or `int64_t` inputs for split");
              }

              if (splitSizes.size() == 0) {
                fail_shape_inference(
                    "Input 'split' can not be empty.");
              }

              const auto& splitDim = inputShape.dim(axis);
              if (!splitDim.has_dim_value()) {
                // Unable to verify nor infer exact split dimension size.
                return -1;
              }

              int64_t splitDimValue = splitDim.dim_value();
              const auto& splitShape = ctx.getInputType(1)->tensor_type().shape();
              if (splitShape.dim_size() == 0) {
                // split is scalar
                if (splitDimValue % splitSizes[0] == 0) {
                  // all output chunks have the same shape, assign that to output sequence shape.
                  return splitSizes[0];
                }
                return -1;
              } else {
                // split is 1-D tensor
                int64_t splitSizesSum = std::accumulate(splitSizes.begin(), splitSizes.end(), (int64_t)0);
                if (splitDimValue != splitSizesSum) {
                  fail_shape_inference(
                      "Sum of split values not equal to 'input' dim size on 'axis'. 'axis' dim size=",
                      splitDimValue,
                      " sum of split values=",
                      splitSizesSum);
                }
                if (std::adjacent_find(splitSizes.begin(), splitSizes.end(), std::not_equal_to<int64_t>()) == splitSizes.end()) {
                  // all split sizes are the same.
                  return splitSizes[0];
                }
                return -1;
              }
            }();
          }

          if (keepdims) {
            auto* outputShape = 
                ctx.getOutputType(0)
                    ->mutable_sequence_type()
                    ->mutable_elem_type()
                    ->mutable_tensor_type()
                    ->mutable_shape();
            *outputShape = inputShape;
            auto* dim = outputShape->mutable_dim(axis);
            // Tensors in sequence could not have different shapes explicitly.
            // Only assign dim_value when all chunks have the same shape.
            if (splitSize > 0) {
              dim->set_dim_value(splitSize);
            } else {
              dim->clear_dim_value();
              dim->clear_dim_param();
            }
          } else {
            TensorShapeProto* outputShape =
                ctx.getOutputType(0)
                    ->mutable_sequence_type()
                    ->mutable_elem_type()
                    ->mutable_tensor_type()
                    ->mutable_shape();
            for (int i = 0; i < inputShape.dim_size(); ++i) {
              if (i != axis) {
                auto* dim = outputShape->add_dim();
                dim->CopyFrom(inputShape.dim(i));
              }
            }
          }
        }));

static const char* ConcatFromSequence_ver11_doc = R"DOC(
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    ConcatFromSequence,
    11,
    OpSchema()
        .Attr(
            "axis",
            "Which axis to concat on. Accepted range in `[-r, r - 1]`, "
            "where `r` is the rank of input tensors. "
            "When `new_axis` is 1, accepted range is `[-r - 1, r]`. ",
            AttributeProto::INT)
        .Attr(
            "new_axis",
            "Insert and concatenate on a new axis or not, "
            "default 0 means do not insert new axis.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .SetDoc(ConcatFromSequence_ver11_doc)
        .Input(
            0,
            "input_sequence",
            "Sequence of tensors for concatenation",
            "S")
        .Output(0, "concat_result", "Concatenated tensor", "T")
        .TypeConstraint(
            "S",
            OpSchema::all_tensor_sequence_types(),
            "Constrain input types to any tensor type.")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto elem_type = 
              ctx.getInputType(0)->sequence_type().elem_type().tensor_type().elem_type();
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(elem_type);

          if (!hasInputShape(ctx, 0)) {
            return;
          }

          auto axis_attr = ctx.getAttribute("axis");
          if (!axis_attr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axis_attr->i());

          int new_axis = 0;
          auto new_axis_attr = ctx.getAttribute("new_axis");
          if (new_axis_attr) {
            new_axis = static_cast<int>(new_axis_attr->i());
          }

          const auto& input_shape =
              ctx.getInputType(0)->sequence_type().elem_type().tensor_type().shape();
          auto rank = input_shape.dim_size();
          if (1 != new_axis && 0 != new_axis) {
            fail_shape_inference("new_axis must be either 0 or 1");
          }

          auto upper_bound = 1 == new_axis ? rank : rank - 1;
          auto lower_bound = 1 == new_axis ? -rank - 1 : -rank;

          if (axis < lower_bound || axis > upper_bound) {
            fail_shape_inference(
              "Invalid value of attribute 'axis'. Accepted range=[",
              lower_bound,
              ", ",
              upper_bound,
              "], Value=",
              axis);
          }

          if (axis < 0) {
            axis += (upper_bound + 1);
          }

          auto* output_shape =
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape();

          for (int i = 0; i <= upper_bound; ++i) {
            output_shape->add_dim();
            if (i != axis) {
              output_shape->mutable_dim(i)->CopyFrom(input_shape.dim((i > axis && new_axis) ? i - 1 : i));
            }
          }
        }));

} // namespace ONNX_NAMESPACE
