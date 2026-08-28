// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/optional/utils.h"

#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::defs::optional::utils {

static constexpr const char* Optional_ver15_doc = R"DOC(
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
)DOC";

std::function<void(OpSchema&)> OptionalOpGenerator(
    std::vector<std::string> tensor_and_sequence_types,
    std::vector<std::string> optional_types) {
  return [tensor_and_sequence_types = std::move(tensor_and_sequence_types),
          optional_types = std::move(optional_types)](OpSchema& schema) {
    schema.SetDoc(Optional_ver15_doc)
        .Input(0, "input", "The input element.", "V", OpSchema::Optional)
        .Attr("type", "Type of the element in the optional output", AttributeProto::TYPE_PROTO, OPTIONAL_VALUE)
        .Output(0, "output", "The optional output enclosing the input element.", "O")
        .TypeConstraint("V", tensor_and_sequence_types, "Constrain input type to all tensor and sequence types.")
        .TypeConstraint("O", optional_types, "Constrain output type to all optional tensor or optional sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("Optional is expected to have an output.");
          }

          const size_t numInputs = ctx.getNumInputs();
          const auto* attr_proto = ctx.getAttribute("type");

          if ((numInputs == 0) && (attr_proto != nullptr)) {
            if (!attr_proto->has_tp())
              fail_type_inference("Attribute 'type' should be a TypeProto and it should specify a type.");
            auto attr_tp = attr_proto->tp();

            ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(attr_tp);
          } else if (numInputs == 1) {
            const auto* input_type = ctx.getInputType(0);
            if (input_type == nullptr) {
              fail_type_inference("Input type is null. Type information is expected for the input.");
            }
            ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type()->CopyFrom(*input_type);
          } else {
            fail_type_inference("Optional is expected to have either an input or the type attribute set.");
          }
        });
  };
}

static constexpr const char* OptionalHasElement_ver18_doc = R"DOC(
Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
)DOC";

std::function<void(OpSchema&)> OptionalHasElementOpGenerator(std::vector<std::string> o_types, bool is_opset18) {
  return [o_types = std::move(o_types), is_opset18](OpSchema& schema) {
    schema.SetDoc(OptionalHasElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O", OpSchema::Optional)
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that optional-type input contains an element. Otherwise, it is empty.",
            "B")
        .TypeConstraint(
            "O",
            o_types,
            is_opset18 ? "Constrain input type to optional tensor and optional sequence types."
                       : "Constrain input type to optional, tensor and sequence types.")
        .TypeConstraint("B", {types::Bool}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 0 && numInputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 0 or 1 input.");
          }
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 output.");
          }
          auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::BOOL);
          output_tensor_type->mutable_shape()->Clear();
        });
  };
}

static constexpr const char* OptionalGetElement_ver18_doc = R"DOC(
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
)DOC";

std::function<void(OpSchema&)> OptionalGetElementOpGenerator(
    std::vector<std::string> optional_types,
    std::vector<std::string> tensor_and_sequence_types,
    bool is_opset18) {
  std::vector<std::string> opt_tensor_seq;
  opt_tensor_seq.insert(opt_tensor_seq.begin(), optional_types.begin(), optional_types.end());
  opt_tensor_seq.insert(opt_tensor_seq.end(), tensor_and_sequence_types.begin(), tensor_and_sequence_types.end());

  return [opt_tensor_seq, tensor_and_sequence_types, is_opset18](OpSchema& schema) {
    schema.SetDoc(OptionalGetElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(0, "output", "Output element in the optional input.", "V")
        .TypeConstraint(
            "O",
            opt_tensor_seq,
            is_opset18 ? "Constrain input type to optional tensor and optional sequence types."
                       : "Constrain input type to optional, tensor and sequence types.")
        .TypeConstraint("V", tensor_and_sequence_types, "Constrain output type to all tensor or sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalGetElement must have an input element.");
          }
          const auto* input_type = ctx.getInputType(0);
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
        });
  };
}
} // namespace ONNX_NAMESPACE::defs::optional::utils
