// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/optional/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {
static std::vector<std::string> tensor_and_sequence_types() {
  return types::Concat(OpSchema::all_tensor_types(), OpSchema::all_tensor_sequence_types());
}

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    15,
    OpSchema()
        .SetDoc(kDoc_OptionalHasElement_ver15)
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
        .TypeConstraint("B", {types::Bool}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 input.");
          }
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 output.");
          }
          auto output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::BOOL);
          output_tensor_type->mutable_shape()->Clear();
        }));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    15,
    OpSchema()
        .SetDoc(kDoc_OptionalGetElement_ver15)
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

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    15,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalOpGenerator(tensor_and_sequence_types(), OpSchema::all_optional_types())));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    18,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalHasElementOpGenerator(
            types::Concat(OpSchema::all_optional_types(), tensor_and_sequence_types()))));

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    OpSchema().FillUsing(
        defs::optional::utils::OptionalGetElementOpGenerator(
            OpSchema::all_optional_types(),
            tensor_and_sequence_types())));

} // namespace ONNX_NAMESPACE
