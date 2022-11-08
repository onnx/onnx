/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include <algorithm>
#include <numeric>

namespace ONNX_NAMESPACE {
static std::vector<std::string> all_map_value_types() {
    auto map_key_types = OpSchema::all_map_key_types();
    // Values have to be one of the following data types
    // TENSOR, SPARSE_TENSOR, MAP, SEQUENCE.
    auto tensor_types = OpSchema::all_tensor_types();
    auto sequence_types = OpSchema::all_tensor_sequence_types();

    static std::vector<std::string> all_map_value_types;
    all_map_value_types.insert(all_map_value_types.end(), tensor_types.begin(), tensor_types.end());
    all_map_value_types.insert(all_map_value_types.end(), sequence_types.begin(), sequence_types.end());
    for (auto key_type : map_key_types) {
        for (auto value_type : tensor_types) {
            std::string map_type = "map("+ key_type + ", " + value_type + ")";
            all_map_value_types.emplace_back(map_type);
        }
    }
    for (auto key_type : map_key_types) {
        for (auto value_type : sequence_types) {
            std::string map_type = "map("+ key_type + ", " + value_type + ")";
            all_map_value_types.emplace_back(map_type);
        }
    }
    return all_map_value_types;
}

static std::vector<std::string> all_map_seq_types() {
    auto map_value_types = all_map_value_types();
    static std::vector<std::string> all_map_seq_types;
    for (auto value_type : map_value_types) {
        std::string seq_map_type = "seq("+ value_type +")";
        all_map_seq_types.emplace_back(seq_map_type);
    }
    return all_map_seq_types;
}

static std::vector<std::string> all_map_types() {
    auto map_key_types = OpSchema::all_map_key_types();
    auto map_value_types = all_map_value_types();
    static std::vector<std::string> all_map_types;
    for (auto key_type : map_key_types) {
        for (auto value_type : map_value_types) {
            std::string map_type = "map("+ key_type + ", " + value_type + ")";
            all_map_types.emplace_back(map_type);
        }
    }
    return all_map_types;
}

static const char* MapConstruct_ver18_doc = R"DOC(
Constructs either an empty map of a certain type specified by the key_type and value_type attributes,
or a map structure with specified 'keys' and 'values'
All 'keys' must have the same data type.
All 'values' must all be of the same type (tensor, sequence or map) and have the same data type.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapConstruct,
    18,
    OpSchema()
        .SetDoc(MapConstruct_ver18_doc)
        .Attr(
            "key_type",
            "The type of the keys present in the map output.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "value_type",
            "The type of the values pairs present in the map output.",
            AttributeProto::TYPE_PROTO,
            OPTIONAL_VALUE)
        .Input(0, "keys", "Keys.", "T", OpSchema::Optional)
        .Input(0, "values", "Sequence enclosing the values.", "S", OpSchema::Optional)
        .Output(0, "map", "Map.", "M")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_tensor_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "S",
            all_map_seq_types(),
            "Constrain input types to any sequence type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {

          const size_t numInputs = ctx.getNumInputs();
          const auto* key_proto = ctx.getAttribute("key_type");
          const auto* value_proto = ctx.getAttribute("value_type");

          if ((numInputs == 0) && (key_proto != nullptr) && (value_proto != nullptr)) {
            if (!key_proto->has_i()) {
              fail_type_inference("Attribute key_type should be of integer type and specify a type.");
            }
            if (!value_proto->has_tp()) {
              fail_type_inference("Attribute value_type should be a TypeProto and it should specify a type.");
            }
            auto key_tp = key_proto->i();
            auto value_tp = value_proto->tp();
            ctx.getOutputType(0)->mutable_map_type()->set_key_type(key_tp);
            ctx.getOutputType(0)->mutable_map_type()->mutable_value_type()->CopyFrom(value_tp);
          } else if (numInputs == 1) {
            fail_type_inference("Only one of 'keys' and 'values' is provided. MapConstruct is expected to have either both inputs or both the type attributes set.");
          } else if (numInputs == 2) {
            auto key_type = ctx.getInputType(0);
            if (key_type == nullptr) {
              fail_type_inference("'Keys' type is null. Type information is expected for this input.");
            }
            auto value_type = ctx.getInputType(1);
            if (value_type == nullptr) {
              fail_type_inference("'Values' type is null. Type information is expected for this input.");
            }
            ctx.getOutputType(0)->mutable_map_type()->set_key_type(key_type->tensor_type().elem_type());
            ctx.getOutputType(0)->mutable_map_type()->mutable_value_type()->CopyFrom(value_type->sequence_type().elem_type());
          } else {
            fail_type_inference("MapConstruct is expected to have either have both inputs or both the type attributes set.");
          }

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapConstruct is expected to have 1 output.");
          }

          // TODO: Shape Inference 
        }));

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
            OpSchema::all_map_key_tensor_types(),
            "Constrain output types to integral and string tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input_type = ctx.getInputType(0);
          if (nullptr == input_type) {
            fail_type_inference("Input map is expected to have type info. Current type is null.");
          }
          const auto key_type = input_type->map_type().key_type();
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(key_type);

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapKeys is expected to have 1 output.");
          }
          // TODO: Shape Inference
        }));


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
            all_map_seq_types(),
            "Constrain output types to any sequence type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input_type = ctx.getInputType(0);
          if (nullptr == input_type) {
            fail_type_inference("Input map is expected to have type info. Current type is null.");
          }
          ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->CopyFrom(input_type->map_type().value_type());

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapValues is expected to have 1 output.");
          }

          // TODO: Shape Inference
        }));

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
        .Input(0, "key", "Key.", "t")
        .Input(0, "value", "Value.", "V")
        .Output(0, "output_map", "Output map that contains the new key value pair.", "M")
        .TypeConstraint(
            "t",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string types.")
        .TypeConstraint(
            "V",
            all_map_value_types(),
            "Constrain input types to any sequence, tensor or map type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input0_type = ctx.getInputType(0);
          const auto input1_type = ctx.getInputType(0);
          const auto input2_type = ctx.getInputType(0);
          if (nullptr == input0_type || nullptr == input1_type || nullptr == input2_type) {
            fail_type_inference("Input map, key and corresponding value are expected to have type info. Current type is null.");
          }

          const auto key_type = input0_type->map_type().key_type();
          const auto value_type = input0_type->map_type().value_type();
          if (key_type != input1_type->tensor_type().elem_type()) {
            fail_type_inference("Input map keys and key to be inserted are expected to have same type. Current types differ.");
          }

          ctx.getOutputType(0)->mutable_map_type()->set_key_type(key_type);
          ctx.getOutputType(0)->mutable_map_type()->mutable_value_type()->CopyFrom(value_type);

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapInsertPair is expected to have 1 output.");
          }

          // TODO: Shape Inference
        }));

static const char* MapDeletePair_ver18_doc = R"DOC(
Delete a 'key'-value pair from the 'map' input.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    MapDeletePair,
    18,
    OpSchema()
        .SetDoc(MapDeletePair_ver18_doc)
        .Input(0, "map", "Input map.", "M")
        .Input(0, "key", "Key.", "t")
        .Output(0, "output_map", "Output map without the provided key value pair.", "M")
        .TypeConstraint(
            "t",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input0_type = ctx.getInputType(0);
          const auto input1_type = ctx.getInputType(0);
          if (nullptr == input0_type || nullptr == input1_type) {
            fail_type_inference("Input map and key corresponding value are expected to have type info. Current type is null.");
          }

          const auto key_type = input0_type->map_type().key_type();
          const auto value_type = input0_type->map_type().value_type();
          if (key_type != input1_type->tensor_type().elem_type()) {
            fail_type_inference("Input map keys and key to be inserted are expected to have same type. Current types differ.");
          }

          ctx.getOutputType(0)->mutable_map_type()->set_key_type(key_type);
          ctx.getOutputType(0)->mutable_map_type()->mutable_value_type()->CopyFrom(value_type);

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapDeletePair is expected to have 1 output.");
          }

          // TODO: Shape Inference
        }));

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
        .Input(0, "key", "Key.", "t")
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that the key is present in the map.",
            "B")
        .TypeConstraint(
            "t",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input0_type = ctx.getInputType(0);
          const auto input1_type = ctx.getInputType(0);
          if (nullptr == input0_type || nullptr == input1_type) {
            fail_type_inference("Input map and key are expected to have type info. Current type is null.");
          }

          const auto key_type = input0_type->map_type().key_type();
          if (key_type != input1_type->tensor_type().elem_type()) {
            fail_type_inference("Input map keys and key to be inserted are expected to have same type. Current types differ.");
          }
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::BOOL);

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapHasKey is expected to have 1 output.");
          }

          // TODO: Shape Inference
        }));

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
        .Output(0, "value", "Value corresponding to the provided key", "V")
        .TypeConstraint(
            "T",
            OpSchema::all_map_key_types(),
            "Constrain input types to integral and string tensor types.")
        .TypeConstraint(
            "V",
            all_map_value_types(),
            "Constrain input types to any sequence, tensor or map type.")
        .TypeConstraint(
            "M",
            all_map_types(),
            "Constrain output types to any map type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const auto input0_type = ctx.getInputType(0);
          const auto input1_type = ctx.getInputType(0);
          if (nullptr == input0_type || nullptr == input1_type) {
            fail_type_inference("Input map and key are expected to have type info. Current type is null.");
          }

          const auto key_type = input0_type->map_type().key_type();
          const auto value_type = input0_type->map_type().value_type();
          if (key_type != input1_type->tensor_type().elem_type()) {
            fail_type_inference("Input map keys and key to be inserted are expected to have same type. Current types differ.");
          }

          ctx.getOutputType(0)->CopyFrom(value_type);

          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("MapGetValue is expected to have 1 output.");
          }

          // TODO: Shape Inference
        }));

} // namespace ONNX_NAMESPACE