/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include "onnx/defs/sequence/utils.h"

#include <algorithm>
#include <numeric>
#include <string>

namespace ONNX_NAMESPACE {
namespace defs {
namespace sequence {
namespace utils {

// Common documentation for SplitToSequence operator, versions 11 and 24
static const char* SplitToSequence_ver11_doc =
    R"DOC(
Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.
)DOC";

std::function<void(OpSchema&)> SplitToSequenceOpGenerator(
    const std::vector<std::string>& input_types,
    const std::vector<std::string>& output_types) {
  return [=](OpSchema& schema) {
    schema.Input(0, "input", "The tensor to split", "T")
        .Input(
            1,
            "split",
            "Length of each output. "
            "It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must be >= 0. ",
            "I",
            OpSchema::Optional)
        .Output(0, "output_sequence", "One or more outputs forming a sequence of tensors after splitting", "S")
        .TypeConstraint("T", input_types, "Constrain input types to all tensor types.")
        .TypeConstraint("I", {"tensor(int32)", "tensor(int64)"}, "Constrain split size to integral tensor.")
        .TypeConstraint("S", output_types, "Constrain output types to all tensor types.")
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
          const auto input0_type = ctx.getInputType(0);
          if (nullptr == input0_type) {
            fail_type_inference("Input type for input at index 0 is null. Type info is expected.")
          }
          ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
              input0_type->tensor_type().elem_type());

          if (!hasInputShape(ctx, 0)) {
            return;
          }

          const auto& inputShape = input0_type->tensor_type().shape();

          int r = inputShape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis > r - 1) {
            fail_shape_inference("Invalid value of attribute 'axis'. Rank=", r, " Value=", axis);
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
              if (!hasInputShape(ctx, 1)) {
                return -1;
              }
              const TensorProto* splitInitializer = ctx.getInputData(1);
              if (nullptr == splitInitializer || !splitInitializer->has_data_type()) {
                return -1;
              }

              std::vector<int64_t> splitSizes;
              if (splitInitializer->data_type() == TensorProto::INT64) {
                const auto data = ParseData<int64_t>(splitInitializer);
                splitSizes.insert(splitSizes.end(), data.begin(), data.end());
              } else if (splitInitializer->data_type() == TensorProto::INT32) {
                const auto data = ParseData<int32_t>(splitInitializer);
                splitSizes.insert(splitSizes.end(), data.begin(), data.end());
              } else {
                // unaccepted data type
                fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for split");
              }

              if (splitSizes.empty()) {
                fail_shape_inference("Input 'split' can not be empty.");
              }

              const auto& splitDim = inputShape.dim(axis);
              if (!splitDim.has_dim_value()) {
                // Unable to verify nor infer exact split dimension size.
                return -1;
              }

              int64_t splitDimValue = splitDim.dim_value();
              const auto& splitShape = getInputShape(ctx, 1);
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
                if (std::adjacent_find(splitSizes.begin(), splitSizes.end(), std::not_equal_to()) == splitSizes.end()) {
                  // all split sizes are the same.
                  return splitSizes[0];
                }
                return -1;
              }
            }();
          }

          if (keepdims) {
            auto* outputShape = ctx.getOutputType(0)
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
            TensorShapeProto* outputShape = ctx.getOutputType(0)
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
        });
  };
}

} // namespace utils
} // namespace sequence
} // namespace defs
} // namespace ONNX_NAMESPACE
