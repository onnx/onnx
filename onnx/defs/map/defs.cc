/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include <algorithm>
#include <numeric>

namespace ONNX_NAMESPACE {
static std::vector<std::string> all_map_types() {
    auto map_key_types = OpSchema::all_map_key_types();
    // Values have to be one of the following data types
    // TENSOR, SPARSE_TENSOR, MAP, SEQUENCE.
    auto tensor_types = OpSchema::all_tensor_types();
    auto sequence_types = OpSchema::all_tensor_sequence_types();

    static std::vector<std::string> all_map_types;
    for (auto key_type : map_key_types) {
        for (auto value_type : tensor_types) {
            std::string map_type = "map("+ key_type + ", " + value_type + ")";
            all_map_types.emplace_back(map_type);
        }
    }
    for (auto key_type : map_key_types) {
        for (auto value_type : sequence_types) {
            std::string map_type = "map("+ key_type + ", " + value_type + ")";
            all_map_types.emplace_back(map_type);
        }
    }
    return all_map_types;
}

static const char* MapEmpty_ver18_doc = R"DOC(
Construct an empty map structure with given data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapEmpty,
    18,
    OpSchema()
        .SetDoc(MapEmpty_ver18_doc)
        .Attr(
            "dtype",
            "(Optional) The data type of the tensors in the output sequence. "
            "The default type is 'map(string, seq(tensor(int64)))'.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Output(0, "map", " Empty map.", "M")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto* attr_proto = ctx.getAttribute("dtype");
          auto elem_type = TensorProto::FLOAT;
          if (nullptr != attr_proto) {
            if (!attr_proto->has_i()) {
              fail_type_inference("Attribute dtype should be of integer type and specify a type.");
            }
            auto attr_value = attr_proto->i();
            elem_type = static_cast<TensorProto_DataType>(attr_value);
          }
          ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
              elem_type);
        })*/);

static const char* MapConstruct_ver18_doc = R"DOC(
Construct a map structure with specified 'keys' and 'values'
All tensors in 'values' must have the same data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapConstruct,
    18,
    OpSchema()
        .SetDoc(MapConstruct_ver18_doc)
        .Input(0, "keys", "Keys.", "T")
        .Input(0, "values", "Sequence enclosing the values.", "S")
        .Output(0, "map", "Map.", "M")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "S",
            all_map_types(),
            "Constrain input types to any sequence type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs < 1) {
            fail_type_inference("SequenceConstruct is expected to have at least 1 input.");
          }

          std::vector<int> input_elem_types;
          input_elem_types.reserve(numInputs);
          for (size_t i = 0; i < numInputs; ++i) {
            auto input_type = ctx.getInputType(i);
            if (nullptr == input_type) {
              fail_type_inference("Input type for input at index ", i, " is null. Type info is expected.");
            }
            input_elem_types.emplace_back(input_type->tensor_type().elem_type());
          }
          if (std::adjacent_find(input_elem_types.begin(), input_elem_types.end(), std::not_equal_to<int>()) !=
              input_elem_types.end()) {
            // not all input elem types are the same.
            fail_type_inference("Element type of inputs are expected to be the same.");
          }

          auto* output_tensor_type =
              ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();

          output_tensor_type->set_elem_type(static_cast<TensorProto_DataType>(input_elem_types[0]));

          if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }

          *(output_tensor_type->mutable_shape()) = ctx.getInputType(0)->tensor_type().shape();

          for (size_t i = 1; i < numInputs; ++i) {
            const auto& input_shape = ctx.getInputType(i)->tensor_type().shape();
            UnionShapeInfo(input_shape, *output_tensor_type);
          }
        })*/);

static const char* MapKeys_ver18_doc = R"DOC(
Outputs a tensor that consists of all the 'keys' present
in the map structure.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapKeys,
    18,
    OpSchema()
        .SetDoc(MapKeys_ver18_doc)
        .Input(0, "map", " Map.", "M")
        .Output(0, "keys", "Keys.", "T")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain input types to any map type.")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain output types to integral and string tensor types.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto* attr_proto = ctx.getAttribute("dtype");
          auto elem_type = TensorProto::FLOAT;
          if (nullptr != attr_proto) {
            if (!attr_proto->has_i()) {
              fail_type_inference("Attribute dtype should be of integer type and specify a type.");
            }
            auto attr_value = attr_proto->i();
            elem_type = static_cast<TensorProto_DataType>(attr_value);
          }
          ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
              elem_type);
        })*/);


static const char* MapValues_ver18_doc = R"DOC(
Outputs a tensor that consists of all the 'values' present
in the map structure.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapValues,
    18,
    OpSchema()
        .SetDoc(MapValues_ver18_doc)
        .Input(0, "map", " Map.", "M")
        .Output(0, "values", "Sequence enclosing the values.", "S")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain input types to any map type.")
        .TypeConstraint(
            "S",
            all_map_types(),
            "Constrain output types to any sequence type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto* attr_proto = ctx.getAttribute("dtype");
          auto elem_type = TensorProto::FLOAT;
          if (nullptr != attr_proto) {
            if (!attr_proto->has_i()) {
              fail_type_inference("Attribute dtype should be of integer type and specify a type.");
            }
            auto attr_value = attr_proto->i();
            elem_type = static_cast<TensorProto_DataType>(attr_value);
          }
          ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
              elem_type);
        })*/);

static const char* MapInsertPair_ver18_doc = R"DOC(
Insert a 'key'-'value' pair into the 'map' input.
'key' and 'value' should have the same type as the corresponding
map's key and value types. If 'key' is already present in the map,
update it to be mapped to the new 'value'.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapInsertPair,
    18,
    OpSchema()
        .SetDoc(MapInsertPair_ver18_doc)
        .Input(0, "map", "Input map.", "M")
        .Input(0, "key", "Key.", "T")
        .Input(0, "value", "Value.", "S")
        .Output(0, "output_map", "Output map that contains the new key value pair.", "M")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "S",
            all_map_types(),
            "Constrain input types to any sequence type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs < 1) {
            fail_type_inference("SequenceConstruct is expected to have at least 1 input.");
          }

          std::vector<int> input_elem_types;
          input_elem_types.reserve(numInputs);
          for (size_t i = 0; i < numInputs; ++i) {
            auto input_type = ctx.getInputType(i);
            if (nullptr == input_type) {
              fail_type_inference("Input type for input at index ", i, " is null. Type info is expected.");
            }
            input_elem_types.emplace_back(input_type->tensor_type().elem_type());
          }
          if (std::adjacent_find(input_elem_types.begin(), input_elem_types.end(), std::not_equal_to<int>()) !=
              input_elem_types.end()) {
            // not all input elem types are the same.
            fail_type_inference("Element type of inputs are expected to be the same.");
          }

          auto* output_tensor_type =
              ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();

          output_tensor_type->set_elem_type(static_cast<TensorProto_DataType>(input_elem_types[0]));

          if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }

          *(output_tensor_type->mutable_shape()) = ctx.getInputType(0)->tensor_type().shape();

          for (size_t i = 1; i < numInputs; ++i) {
            const auto& input_shape = ctx.getInputType(i)->tensor_type().shape();
            UnionShapeInfo(input_shape, *output_tensor_type);
          }
        })*/);

static const char* MapDeletePair_ver18_doc = R"DOC(
Delete a 'key'-value pair from the 'map' input.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapDeletePair,
    18,
    OpSchema()
        .SetDoc(MapDeletePair_ver18_doc)
        .Input(0, "map", "Input map.", "M")
        .Input(0, "key", "Key.", "T")
        .Output(0, "output_map", "Output map without the provided key value pair.", "M")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs < 1) {
            fail_type_inference("SequenceConstruct is expected to have at least 1 input.");
          }

          std::vector<int> input_elem_types;
          input_elem_types.reserve(numInputs);
          for (size_t i = 0; i < numInputs; ++i) {
            auto input_type = ctx.getInputType(i);
            if (nullptr == input_type) {
              fail_type_inference("Input type for input at index ", i, " is null. Type info is expected.");
            }
            input_elem_types.emplace_back(input_type->tensor_type().elem_type());
          }
          if (std::adjacent_find(input_elem_types.begin(), input_elem_types.end(), std::not_equal_to<int>()) !=
              input_elem_types.end()) {
            // not all input elem types are the same.
            fail_type_inference("Element type of inputs are expected to be the same.");
          }

          auto* output_tensor_type =
              ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();

          output_tensor_type->set_elem_type(static_cast<TensorProto_DataType>(input_elem_types[0]));

          if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) {
            return;
          }

          *(output_tensor_type->mutable_shape()) = ctx.getInputType(0)->tensor_type().shape();

          for (size_t i = 1; i < numInputs; ++i) {
            const auto& input_shape = ctx.getInputType(i)->tensor_type().shape();
            UnionShapeInfo(input_shape, *output_tensor_type);
          }
        })*/);

static const char* MapHasKey_ver18_doc = R"DOC(
Returns true if the 'key' is present in the 'map' input.
Otherwise this op returns false.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapHasKey,
    18,
    OpSchema()
        .SetDoc(MapHasKey_ver18_doc)
        .Input(0, "map", "Input map.", "M")
        .Input(0, "key", "Key.", "T")
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that the key is present in the map.",
            "B")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        })*/);

static const char* MapGetValue_ver18_doc = R"DOC(
Returns the 'value' associated with the provided 'key' present in the 'map' input.
It is an error if the 'key' is not present in the 'map.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapGetValue,
    18,
    OpSchema()
        .SetDoc(MapGetValue_ver18_doc)
        .Input(0, "map", "Input map.", "M")
        .Input(0, "key", "Key for which the value is retrieved.", "T")
        .Output(0, "value", "Value corresponding to the provided key", "S")
        .Output(0, "map", "Map.", "M")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "S",
            all_map_types(),
            "Constrain input types to any sequence type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        /*.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        })*/);

} // namespace ONNX_NAMESPACE