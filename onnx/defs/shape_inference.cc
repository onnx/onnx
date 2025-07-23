/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shape_inference.h"

#include <vector>

#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/onnx_pb.h"
#include "onnx/defs/schema.h"
#include "onnx/common/ir_pb_converter.h"
#include <unordered_map>
#include <unordered_set>
namespace ONNX_NAMESPACE {

// Note: for all methods below for propagating type or shape, callers are
// responsible to handle optional inputs/outputs and ensure that the specified
// index value is less than NumInputs/NumOutputs.
// Supports mixed tensor and sparse tensor
void propagateElemTypeFromTensorInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  const auto input_value_case = input_type->value_case();
  if (input_value_case != TypeProto::kTensorType && input_value_case != TypeProto::kSparseTensorType) {
    fail_type_inference(
        "Input ", inputIndex, " expected to have tensor or sparse tensor type. Got: ", input_value_case);
  }

  const auto input_elem_type = getTensorElementType(*input_type);
  if (input_elem_type == TensorProto::UNDEFINED) {
    fail_type_inference("Element type of input ", inputIndex, " unknown");
  }
  auto output_type = ctx.getOutputType(outputIndex);
  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    setTensorElementType(input_elem_type, output_value_case, *output_type);
  } else if (output_value_case == TypeProto::VALUE_NOT_SET) {
    // Assume output will have the same type
    setTensorElementType(input_elem_type, input_value_case, *output_type);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor or sparse tensor type. Got: ", output_value_case);
  }
}

static void propagateElemTypeFromSequenceInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kSequenceType) {
    fail_type_inference("Input ", inputIndex, " expected to have sequence type");
  }
  auto input_seq_type = input_type->sequence_type();
  if (!input_seq_type.has_elem_type()) {
    fail_type_inference("Element type of sequence input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_sequence_type()->mutable_elem_type()->CopyFrom(input_seq_type.elem_type());
}

static void propagateElemTypeFromOptionalInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kOptionalType) {
    fail_type_inference("Input ", inputIndex, " expected to have optional type");
  }
  auto input_opt_type = input_type->optional_type();
  if (!input_opt_type.has_elem_type()) {
    fail_type_inference("Element type of optional input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_optional_type()->mutable_elem_type()->CopyFrom(input_opt_type.elem_type());
}

static void propagateElemTypeFromMapInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type || input_type->value_case() != TypeProto::kMapType) {
    fail_type_inference("Input ", inputIndex, " expected to have map type");
  }
  auto input_map_type = input_type->map_type();
  if (!input_map_type.has_key_type()) {
    fail_type_inference("Key type of map input ", inputIndex, " unknown");
  }
  if (!input_map_type.has_value_type()) {
    fail_type_inference("Value type of map input ", inputIndex, " unknown");
  }

  auto output_type = ctx.getOutputType(outputIndex);
  output_type->mutable_map_type()->set_key_type(input_map_type.key_type());
  output_type->mutable_map_type()->mutable_value_type()->CopyFrom(input_map_type.value_type());
}

void propagateElemTypeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input ", inputIndex, " expected to have type but instead is null");
  }
  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    propagateElemTypeFromTensorInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kSequenceType) {
    propagateElemTypeFromSequenceInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kOptionalType) {
    propagateElemTypeFromOptionalInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kMapType) {
    propagateElemTypeFromMapInputToOutput(ctx, inputIndex, outputIndex);
  }
}

/*
Merge shape information from a source shape into a target shape.
* merges each TensorShapeProto_Dimension separately.
* prefer values over params.
* If both have values, values must match.
* prefer target param over source param if mismatched.
* Fail if there are mismatches in number of dimensions or dimension values.
*/
static void mergeInShapeInfo(const TensorShapeProto& source, TensorShapeProto& target) {
  auto num_source_dims = source.dim_size();
  auto num_target_dims = target.dim_size();
  if (num_source_dims != num_target_dims) {
    fail_shape_inference(
        "Mismatch between number of inferred and declared dimensions. inferred=",
        num_source_dims,
        " declared=",
        num_target_dims);
  }

  auto& source_dims = source.dim();
  auto* target_dims = target.mutable_dim();

  for (int i = 0, end = source_dims.size(); i < end; ++i) {
    auto& source_dim = source_dims.Get(i);
    auto& target_dim = *target_dims->Mutable(i);
    mergeInDimensionInfo(source_dim, target_dim, i);
  }
}

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type) {
  if (target_type.has_shape()) {
    // merge with existing info.
    mergeInShapeInfo(source_shape, *target_type.mutable_shape());
  } else {
    // copy to target
    (*target_type.mutable_shape()) = source_shape;
  }
}

void mergeInShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type) {
  if (target_type.has_shape()) {
    // merge with existing info.
    mergeInShapeInfo(source_shape, *target_type.mutable_shape());
  } else {
    // copy to target
    (*target_type.mutable_shape()) = source_shape;
  }
}

/*
Merge the shape information from two TypeProto_Tensor instances.
Values are merged into target from source.
If target has no shape information, copy from source.
If source has no shape information, ignore source.
If both have shape information:
- merge each TensorShapeProto_Dimension separately.
- Prefer values over params. If both have values, values must match.
- Prefer target param over source param if mismatched.
Fail if there are mismatches in number of dimensions or dimension values.
*/
void mergeInShapeInfo(const TypeProto_Tensor& source, TypeProto_Tensor& target) {
  if (source.has_shape())
    mergeInShapeInfo(source.shape(), target);
}

void mergeInShapeInfo(const TypeProto_SparseTensor& source, TypeProto_SparseTensor& target) {
  if (source.has_shape())
    mergeInShapeInfo(source.shape(), target);
}

/// <summary>
/// Utility function for UnionShapeInfoForTensor.
/// Both shapes must be of the same rank
/// </summary>
/// <param name="source_shape"></param>
/// <param name="target_shape">destination shape</param>
static void UnionShapeInfo(const TensorShapeProto& source_shape, TensorShapeProto& target_shape) {
  auto source_rank = source_shape.dim_size();
  for (int i = 0; i < source_rank; ++i) {
    const auto& source_dim = source_shape.dim(i);
    const auto target_dim = target_shape.dim(i);
    bool is_dims_conflict = [&]() {
      if (source_dim.has_dim_value()) {
        return !target_dim.has_dim_value() || target_dim.dim_value() != source_dim.dim_value();
      }

      if (source_dim.has_dim_param()) {
        return !(target_dim.has_dim_param() && target_dim.dim_param() == source_dim.dim_param());
      }

      return (target_dim.has_dim_value() || target_dim.has_dim_param());
    }();
    if (is_dims_conflict && (target_dim.has_dim_value() || target_dim.has_dim_param())) {
      auto dim = target_shape.mutable_dim(i);
      dim->clear_dim_value();
      dim->clear_dim_param();
    }
  }
}

template <typename TENSOR_TYPE>
static void UnionShapeInfoForTensor(const TensorShapeProto& source_shape, TENSOR_TYPE& target_type) {
  if (target_type.has_shape()) {
    TensorShapeProto* target_shape = target_type.mutable_shape();

    auto source_rank = source_shape.dim_size();
    auto target_rank = target_shape->dim_size();
    if (source_rank != target_rank) {
      target_type.clear_shape();
      return;
    }

    UnionShapeInfo(source_shape, *target_shape);
  }
}

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_Tensor& target_type) {
  UnionShapeInfoForTensor(source_shape, target_type);
}

static void UnionShapeInfo(const TypeProto_Tensor& source_type, TypeProto_Tensor& target_type) {
  // The union of a tensor of unknown rank and a tensor of known rank is a tensor of unknown rank.
  // Hence, if the source_type had unknown rank, we clear the shape of the target_type.
  // Otherwise, UnionShapeInfoForTensor handles the rest.
  if (source_type.has_shape()) {
    UnionShapeInfoForTensor(source_type.shape(), target_type);
  } else {
    target_type.clear_shape();
  }
}

static void UnionShapeInfo(const TypeProto_SparseTensor& source_type, TypeProto_SparseTensor& target_type) {
  // The union of a tensor of unknown rank and a tensor of known rank is a tensor of unknown rank.
  // Hence, if the source_type had unknown rank, we clear the shape of the target_type.
  // Otherwise, UnionShapeInfoForTensor handles the rest.
  if (source_type.has_shape()) {
    UnionShapeInfoForTensor(source_type.shape(), target_type);
  } else {
    target_type.clear_shape();
  }
}

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type) {
  UnionShapeInfoForTensor(source_shape, target_type);
}

void UnionTypeInfo(const TypeProto& source_type, TypeProto& target_type) {
  if (source_type.value_case() != target_type.value_case()) {
    fail_type_inference(
        "Mismatched type:", " inferred=", source_type.value_case(), " declared=", target_type.value_case());
  }

  const auto target_case = target_type.value_case();
  if (target_case == TypeProto::ValueCase::kTensorType) {
    auto source_elem_type = source_type.tensor_type().elem_type();
    auto target_elem_type = target_type.tensor_type().elem_type();

    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched tensor element type:",
          " inferred=",
          Utils::DataTypeUtils::ToDataTypeString(source_elem_type),
          " declared=",
          Utils::DataTypeUtils::ToDataTypeString(target_elem_type));
    }

    UnionShapeInfo(source_type.tensor_type(), *target_type.mutable_tensor_type());
  } else if (target_case == TypeProto::ValueCase::kSparseTensorType) {
    auto source_elem_type = source_type.sparse_tensor_type().elem_type();
    auto target_elem_type = target_type.sparse_tensor_type().elem_type();
    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched sparse tensor element type:",
          " inferred=",
          Utils::DataTypeUtils::ToDataTypeString(source_elem_type),
          " declared=",
          Utils::DataTypeUtils::ToDataTypeString(target_elem_type));
    }
    UnionShapeInfo(source_type.sparse_tensor_type(), *target_type.mutable_sparse_tensor_type());
  } else if (target_case == TypeProto::ValueCase::kSequenceType) {
    if (!source_type.sequence_type().has_elem_type()) {
      fail_type_inference("source sequence type missing element type.");
    }
    if (!target_type.sequence_type().has_elem_type()) {
      fail_type_inference("target sequence type missing element type.");
    }
    UnionTypeInfo(source_type.sequence_type().elem_type(), *target_type.mutable_sequence_type()->mutable_elem_type());
  } else if (target_case == TypeProto::ValueCase::kOptionalType) {
    if (!source_type.optional_type().has_elem_type()) {
      fail_type_inference("source optional type missing element type.");
    }
    if (!target_type.optional_type().has_elem_type()) {
      fail_type_inference("target optional type missing element type.");
    }
    UnionTypeInfo(source_type.optional_type().elem_type(), *target_type.mutable_optional_type()->mutable_elem_type());
  } else if (target_case == TypeProto::ValueCase::kMapType) {
    if (!source_type.map_type().has_key_type()) {
      fail_type_inference("source map type missing key type.");
    }
    if (!target_type.map_type().has_key_type()) {
      fail_type_inference("target map type missing key type.");
    }
    auto source_key_type = source_type.map_type().key_type();
    auto target_key_type = target_type.map_type().key_type();
    if (source_key_type != target_key_type) {
      fail_type_inference(
          "Mismatched map tensor key type:",
          " inferred=",
          Utils::DataTypeUtils::ToDataTypeString(source_key_type),
          " declared=",
          Utils::DataTypeUtils::ToDataTypeString(target_key_type));
    }

    if (!source_type.map_type().has_value_type()) {
      fail_type_inference("source map type missing value type.");
    }
    if (!target_type.map_type().has_value_type()) {
      fail_type_inference("target map type missing value type.");
    }
    UnionTypeInfo(source_type.map_type().value_type(), *target_type.mutable_map_type()->mutable_value_type());
  }
}

// Supports both Tensor and SparseTensor
// This does not fail if input_type is Tensor and output type is SparseTensor
// or the other way around. This is to support mixed cases when an op receives
// sparse input and outputs dense or vice-versa.
// If the output value_case is not set, then
// the input value_case is propagated.
static void propagateTensorElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  int32_t input_elem_type = TensorProto::UNDEFINED;
  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    input_elem_type = getTensorElementType(*input_type);
    if (input_elem_type == TensorProto::UNDEFINED) {
      fail_type_inference("Element type of tensor or sparse tensor input was unknown");
    }
  } else {
    fail_type_inference("Input was expected to have tensor or sparse tensor type. Got ", input_value_case);
  }

  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::VALUE_NOT_SET) {
    setTensorElementType(input_elem_type, input_value_case, *output_type);
  } else if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    const auto output_elem_type = getTensorElementType(*output_type);
    if (output_elem_type != TensorProto::UNDEFINED) {
      if (input_elem_type != output_elem_type) {
        fail_type_inference(
            "Input element type of ", input_elem_type, " does not match existing output type of ", output_elem_type);
      }
    } else {
      setTensorElementType(input_elem_type, output_value_case, *output_type);
    }
  } else {
    // This is not expected to happen
    fail_type_inference("Output was expected to have tensor type. Got ", output_value_case);
  }
}

static void propagateSequenceElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kSequenceType) {
    fail_type_inference("Input was expected to have sequence type. Got ", input_type->value_case());
  }

  const auto& input_seq_type = input_type->sequence_type();

  if (input_seq_type.has_elem_type()) {
    propagateElemTypeWithValidation(
        &input_seq_type.elem_type(), output_type->mutable_sequence_type()->mutable_elem_type());
  } else {
    fail_type_inference("Element type of sequence input was unknown");
  }
}

static void propagateOptionalElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kOptionalType) {
    fail_type_inference("Input was expected to have optional type. Got ", input_type->value_case());
  }

  const auto& input_opt_type = input_type->optional_type();

  if (input_opt_type.has_elem_type()) {
    propagateElemTypeWithValidation(
        &input_opt_type.elem_type(), output_type->mutable_optional_type()->mutable_elem_type());
  } else {
    fail_type_inference("Element type of optional input was unknown");
  }
}

static void propagateMapElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kMapType) {
    fail_type_inference("Input was expected to have map type. Got ", input_type->value_case());
  }

  const auto& input_map_type = input_type->map_type();

  if (!input_map_type.has_key_type()) {
    fail_type_inference("Key type of map input was unknown");
  }
  if (!input_map_type.has_value_type()) {
    fail_type_inference("Value type of map input was unknown");
  }
  output_type->mutable_map_type()->set_key_type(input_map_type.key_type());
  propagateElemTypeWithValidation(&input_map_type.value_type(), output_type->mutable_map_type()->mutable_value_type());
}

// propagate the element type from an input type to an output type.
// if an existing output element type exists, validate it matches.
void propagateElemTypeWithValidation(const TypeProto* input_type, TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    propagateTensorElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kSequenceType) {
    propagateSequenceElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kOptionalType) {
    propagateOptionalElemTypeWithValidation(input_type, output_type);
  } else if (input_value_case == TypeProto::kMapType) {
    propagateMapElemTypeWithValidation(input_type, output_type);
  } else {
    fail_type_inference(
        "Input was expected to have either tensor, sequence, optional or map type. Got ", input_value_case);
  }
}

TensorShapeProto getShapeInput(const InferenceContext& ctx, size_t input_index, bool& found) {
  return getShapeInput(ctx, input_index, false, found);
}

TensorShapeProto
getShapeInput(const InferenceContext& ctx, size_t input_index, bool fail_if_negative_value, bool& found) {
  TensorShapeProto shape_input;

  found = false;

  // First, check initializer.
  const TensorProto* shape_initializer = ctx.getInputData(input_index);
  if (shape_initializer) {
    const std::vector<int64_t> shape_data = ParseData<int64_t>(shape_initializer);
    for (const int64_t& e : shape_data) {
      shape_input.add_dim()->set_dim_value(e);
    }
    found = true;
  }

  // Then, check symbolic input.
  const TensorShapeProto* symbolic_input = ctx.getSymbolicInput(input_index);
  if (!found && symbolic_input) {
    shape_input.CopyFrom(*symbolic_input);
    found = true;
  }

  // Try rank inference.
  if (!found && hasInputShape(ctx, input_index)) {
    const TensorShapeProto& shape_input_shape = getInputShape(ctx, input_index);
    if (shape_input_shape.dim_size() != 1) {
      fail_shape_inference("shape input must be 1D tensor");
    }
    if (shape_input_shape.dim(0).has_dim_value()) {
      // Attempt rank inference using shape of shape input
      int64_t dim_value = shape_input_shape.dim(0).dim_value();
      for (int64_t i = 0; i < dim_value; ++i) {
        shape_input.add_dim();
      }
      found = true;
    }
  }

  if (found && fail_if_negative_value) {
    int dims_size = shape_input.dim_size();
    for (int i = 0; i < dims_size; ++i) {
      const auto& dim = shape_input.dim(i);
      if (dim.has_dim_value() && dim.dim_value() < 0) {
        fail_shape_inference("shape input tensor must have non-negative elements");
      }
    }
  }

  return shape_input;
}

template <typename Container>
std::string stringify(const Container& elements) {
  std::stringstream ss;
  for (const auto& element : elements) {
    ss << element << ", ";
  }
  return ss.str();
}

std::pair<int, int> getAttributeProtoElemTypeAndLength(const AttributeProto* attr_proto) {
  if (attr_proto->ints_size()) {
    return {TensorProto_DataType_INT64, attr_proto->ints_size()};
  } else if (attr_proto->floats_size()) {
    return {TensorProto_DataType_FLOAT, attr_proto->floats_size()};
  } else if (attr_proto->strings_size()) {
    return {TensorProto_DataType_STRING, attr_proto->strings_size()};
  } else if (attr_proto->has_t()) {
    if (attr_proto->t().dims_size() != 1) {
      fail_type_inference(
          "Attribute ", attr_proto->name(), " expected to be a 1D tensor but was ", attr_proto->t().dims_size(), "D");
    }
    return {attr_proto->t().data_type(), attr_proto->t().dims(0)};
  }
  return {TensorProto::UNDEFINED, 0};
}

std::pair<int, int> getAttributeElementTypeAndLength(
    const InferenceContext& ctx,
    const std::initializer_list<std::string>& attribute_names) {
  // Get element type and lengths of 1D attribute lists
  int32_t elem_type = TensorProto::UNDEFINED;
  int32_t length = 0;
  for (const auto& attribute : attribute_names) {
    const AttributeProto* attr_proto = ctx.getAttribute(attribute);
    if (attr_proto != nullptr) {
      if (elem_type != TensorProto::UNDEFINED) {
        // Another attribute was already set
        fail_shape_inference("One and only one attribute must be set out of ", stringify(attribute_names));
      }
      std::tie(elem_type, length) = getAttributeProtoElemTypeAndLength(attr_proto);
    }
  }
  return {elem_type, length};
}


// Helper class for type-only inference context
class TypeInferenceContext : public InferenceContext {
private:
    const NodeProto& node_;
    std::vector<const TypeProto*> input_types_;
    std::vector<TypeProto> output_types_;
    std::unordered_map<std::string, const AttributeProto*> attributes_;
    bool data_prop_;

public:
    TypeInferenceContext(
        const NodeProto& node,
        const std::unordered_map<std::string, TypeProto>& value_types,
        bool data_prop = false)
        : node_(node), data_prop_(data_prop) {
        
        // Set up input types
        input_types_.reserve(node.input_size());
        for (const auto& input_name : node.input()) {
            auto it = value_types.find(input_name);
            input_types_.push_back(it != value_types.end() ? &it->second : nullptr);
        }
        
        // Initialize output types
        output_types_.resize(node.output_size());
        
        // Set up attributes map
        for (const auto& attr : node.attribute()) {
            attributes_[attr.name()] = &attr;
        }
    }

    // Override InferenceContext methods
    const TypeProto* getInputType(size_t index) const override {
        return (index < input_types_.size()) ? input_types_[index] : nullptr;
    }
    
    TypeProto* getOutputType(size_t index) override {
        return (index < output_types_.size()) ? &output_types_[index] : nullptr;
    }
    
    const AttributeProto* getAttribute(const std::string& name) const override {
        auto it = attributes_.find(name);
        return (it != attributes_.end()) ? it->second : nullptr;
    }
    
    size_t getNumInputs() const override {
        return node_.input_size();
    }
    
    size_t getNumOutputs() const override {
        return node_.output_size();
    }
    
    // For type-only inference, these shape-related methods return nullptr/false
    const TensorShapeProto* getInputShape(size_t /*index*/) const override {
        return nullptr;
    }
    
    const TensorProto* getInputData(size_t /*index*/) const override {
        return nullptr; // No constant folding for type-only inference
    }
    
    const SparseTensorProto* getInputSparseData(size_t /*index*/) const override {
        return nullptr;
    }
    
    const TensorShapeProto* getSymbolicInput(size_t /*index*/) const override {
        return nullptr;
    }
    
    // Get the inferred output types
    const std::vector<TypeProto>& getOutputTypes() const {
        return output_types_;
    }
};

// Core type inference function for a single node
void InferTypesForNode(
    const NodeProto& node,
    const std::unordered_map<std::string, TypeProto>& input_types,
    std::unordered_map<std::string, TypeProto>& output_types,
    bool data_prop) {
    
    // Get the operator schema
    const auto* schema = OpSchemaRegistry::Schema(node.op_type(), node.domain());
    if (!schema) {
        // Unknown operator - skip type inference
        return;
    }
    
    // Check if operator has type and shape inference function
    if (!schema->has_type_and_shape_inference_function()) {
        return;
    }
    
    try {
        // Create type inference context
        TypeInferenceContext ctx(node, input_types, data_prop);
        
        // Call the operator's inference function
        // Note: This may fail for shape inference, but type parts should still work
        schema->GetTypeAndShapeInferenceFunction()(ctx);
        
        // Extract successfully inferred output types
        const auto& inferred_outputs = ctx.getOutputTypes();
        for (size_t i = 0; i < node.output_size() && i < inferred_outputs.size(); ++i) {
            const std::string& output_name = node.output(i);
            if (inferred_outputs[i].value_case() != TypeProto::VALUE_NOT_SET) {
                output_types[output_name] = inferred_outputs[i];
            } else {
              LOGS_DEFAULT(WARNING) << "Shape inference failed for output " << i 
                                    << " (" << output_name << ") of node " << node.name() 
                                    << " (op_type: " << node.op_type() << ")";
            }
        }
        
    } catch (const std::exception& e) {
        // Type inference failed for this node - continue with others
        // In strict mode, this would be re-thrown by the caller
    }
}

// Main type inference function for a model
void InferTypes(
    ModelProto& model,
    bool check_type,
    bool strict_mode,
    bool data_prop) {
    
    GraphProto* graph = model.mutable_graph();
    
    // Build initial type map from graph inputs and initializers
    std::unordered_map<std::string, TypeProto> value_types;
    
    // Add graph input types
    for (const auto& input : graph->input()) {
        if (input.has_type()) {
            value_types[input.name()] = input.type();
        }
    }
    
    // Add initializer types (infer from tensor data)
    for (const auto& initializer : graph->initializer()) {
        TypeProto type;
        type.mutable_tensor_type()->set_elem_type(initializer.data_type());
        
        // Set shape if available
        TensorShapeProto* shape = type.mutable_tensor_type()->mutable_shape();
        for (int64_t dim_size : initializer.dims()) {
            shape->add_dim()->set_dim_value(dim_size);
        }
        
        value_types[initializer.name()] = type;
    }
    
    // Add existing value_info types
    for (const auto& value_info : graph->value_info()) {
        if (value_info.has_type()) {
            value_types[value_info.name()] = value_info.type();
        }
    }
    
    // Process nodes in topological order
    std::unordered_set<std::string> failed_nodes;
    
    for (const auto& node : graph->node()) {
        try {
            std::unordered_map<std::string, TypeProto> node_output_types;
            InferTypesForNode(node, value_types, node_output_types, data_prop);
            
            // Update global value_types with successfully inferred outputs
            for (const auto& [name, type] : node_output_types) {
                value_types[name] = type;
            }
            
        } catch (const std::exception& e) {
            failed_nodes.insert(node.name());
            
            if (strict_mode) {
                fail_type_inference("Type inference failed for node '", node.name(), "': ", e.what());
            }
            // In non-strict mode, continue with next node
        }
    }
    
    // Update graph's value_info with inferred types
    std::unordered_set<std::string> existing_value_info;
    for (const auto& value_info : graph->value_info()) {
        existing_value_info.insert(value_info.name());
    }
    
    for (const auto& [name, type] : value_types) {
        // Skip if already exists in value_info, inputs, or outputs
        if (existing_value_info.count(name)) {
            continue;
        }
        
        bool is_input = false;
        for (const auto& input : graph->input()) {
            if (input.name() == name) {
                is_input = true;
                break;
            }
        }
        if (is_input) continue;
        
        bool is_output = false;
        for (const auto& output : graph->output()) {
            if (output.name() == name) {
                is_output = true;
                break;
            }
        }
        if (is_output) continue;
        
        bool is_initializer = false;
        for (const auto& initializer : graph->initializer()) {
            if (initializer.name() == name) {
                is_initializer = true;
                break;
            }
        }
        if (is_initializer) continue;
        
        // Add to value_info
        ValueInfoProto* new_value_info = graph->add_value_info();
        new_value_info->set_name(name);
        new_value_info->mutable_type()->CopyFrom(type);
    }
    
    // Type checking if requested
    if (check_type) {
        ValidateTypeConsistency(*graph, value_types);
    }
}

// Path-based type inference function
void InferTypesFromPath(
    const std::string& model_path,
    const std::string& output_path,
    bool check_type,
    bool strict_mode,
    bool data_prop) {
    
    // Load model
    ModelProto model;
    std::fstream model_file(model_path, std::ios::in | std::ios::binary);
    if (!model_file.good() || !model.ParseFromIstream(&model_file)) {
        fail_type_inference("Failed to load model from path: ", model_path);
    }
    model_file.close();
    
    // Perform type inference
    InferTypes(model, check_type, strict_mode, data_prop);
    
    // Save model
    const std::string& actual_output_path = output_path.empty() ? model_path : output_path;
    std::fstream output_file(actual_output_path, std::ios::out | std::ios::binary);
    if (!output_file.good() || !model.SerializeToOstream(&output_file)) {
        fail_type_inference("Failed to save model to path: ", actual_output_path);
    }
    output_file.close();
}

// Helper function to validate type consistency
void ValidateTypeConsistency(
    const GraphProto& graph,
    const std::unordered_map<std::string, TypeProto>& value_types) {
    
    // Check that graph outputs have consistent types
    for (const auto& output : graph.output()) {
        if (output.has_type()) {
            auto it = value_types.find(output.name());
            if (it != value_types.end()) {
                try {
                    TypeProto expected_type = output.type();
                    UnionTypeInfo(it->second, expected_type);
                } catch (const std::exception& e) {
                    fail_type_inference("Output type mismatch for '", output.name(), "': ", e.what());
                }
            }
        }
    }
    
    // Additional consistency checks can be added here
}

} // namespace ONNX_NAMESPACE
