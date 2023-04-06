/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include <algorithm>
#include <numeric>

namespace ONNX_NAMESPACE {

static std::vector<std::string> optional_and_tensor_types() {
  auto optional_types = OpSchema::all_optional_types();
  auto tensor_types = OpSchema::all_tensor_types();
  auto sequence_types = OpSchema::all_tensor_sequence_types();
  optional_types.insert(optional_types.end(), tensor_types.begin(), tensor_types.end());
  optional_types.insert(optional_types.end(), sequence_types.begin(), sequence_types.end());
  return optional_types;
}

static const char* OptionalHasElement_ver1_doc = R"DOC(
Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    15,
    OpSchema()
        .SetDoc(OptionalHasElement_ver1_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that optional-type input contains an element. Otherwise, it is empty.",
            "B")
        .TypeConstraint(
            "O",
            OpSchema::all_optional_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 input.");
          }
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 output.");
          }
          auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::BOOL);
          output_tensor_type->mutable_shape()->Clear();
        }));

static const char* OptionalGetElement_ver1_doc = R"DOC(
Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    15,
    OpSchema()
        .SetDoc(OptionalGetElement_ver1_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(0, "output", "Output element in the optional input.", "V")
        .TypeConstraint(
            "O",
            OpSchema::all_optional_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain output type to all tensor or sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalGetElement must have an input element.");
          }
          auto input_type = ctx.getInputType(0);
          if (input_type == nullptr) {
            fail_type_inference("Input type is null. Input must have Type information.");
          }
          if (!input_type->has_optional_type() || !input_type->optional_type().has_elem_type()) {
            fail_type_inference("Input must be an optional-type value containing an element with type information.");
          }
          ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
        }));

static const char* OptionalGetElement_ver18_doc = R"DOC(
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    OpSchema()
        .SetDoc(OptionalGetElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(0, "output", "Output element in the optional input.", "V")
        .TypeConstraint(
            "O",
            optional_and_tensor_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain output type to all tensor or sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalGetElement must have an input element.");
          }
          auto input_type = ctx.getInputType(0);
          if (input_type == nullptr) {
            fail_type_inference("Input type is null. Input must have Type information.");
          }
          if (input_type->has_optional_type()) {
            if (!input_type->optional_type().has_elem_type()) {
              fail_type_inference("Optional-type input must contain an element with type information.");
            }
            ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
          } else {
            propagateShapeAndTypeFromFirstInput(ctx);
          }
        }));

} // namespace ONNX_NAMESPACE
