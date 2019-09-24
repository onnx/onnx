#pragma once

#include "onnx/defs/data_type_utils.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using Dim = TensorShapeProto_Dimension;

class GraphInferencer {
 public:
  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  virtual std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) = 0;
  virtual ~GraphInferencer() = default;
};

// Exception class used for handling errors in type and shape inference

class InferenceError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  InferenceError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = ONNX_NAMESPACE::MakeString(
        std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_type_inference(...)        \
  throw ONNX_NAMESPACE::InferenceError( \
      ONNX_NAMESPACE::MakeString("[TypeInferenceError] ", __VA_ARGS__));

#define fail_shape_inference(...)       \
  throw ONNX_NAMESPACE::InferenceError( \
      ONNX_NAMESPACE::MakeString("[ShapeInferenceError] ", __VA_ARGS__));

struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual const TensorProto* getInputData(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto* getOutputType(size_t index) = 0;
  virtual GraphInferencer* getGraphAttributeInferencer(
      const std::string& attribute_name) = 0;
  virtual ~InferenceContext() {}
};

using InferenceFunction = std::function<void(InferenceContext&)>;

// This no-op inference function is used for operators without an
// inference implementation.
inline void dummyInferenceFunction(InferenceContext&){};

template <typename T>
inline bool getRepeatedAttribute(
    InferenceContext& ctx,
    std::string attr_name,
    std::vector<T>& values) {
  const auto* attr = ctx.getAttribute(attr_name);
  if (attr) {
    values = RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

inline int64_t getAttribute(
    InferenceContext& ctx,
    const std::string& attributeName,
    int64_t defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_i())
    return attr_proto->i();
  return defaultValue;
}

inline std::string getAttribute(
    InferenceContext& ctx,
    const std::string& attributeName,
    const std::string& defaultValue) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr != attr_proto) && attr_proto->has_s())
    return attr_proto->s();
  return defaultValue;
}

inline TensorShapeProto::Dimension operator*(
    TensorShapeProto::Dimension dim1,
    TensorShapeProto::Dimension dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value() && dim2.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() * dim2.dim_value());
  } else if (dim1.has_dim_value() && (dim1.dim_value() == 1)) {
    return dim2;
  } else if (dim2.has_dim_value() && (dim2.dim_value() == 1)) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator*(
    TensorShapeProto::Dimension dim1,
    int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() * dim2);
  } else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

inline TensorShapeProto::Dimension operator/(
    TensorShapeProto::Dimension dim1,
    int64_t dim2) {
  TensorShapeProto::Dimension result;
  if (dim1.has_dim_value()) {
    result.set_dim_value(dim1.dim_value() / dim2);
  } else if (dim2 == 1) {
    return dim1;
  }
  return result;
}

// if from >= upto_exclusive, return 1.
// Caller must make sure upto_exclusive is less than or equal to shape.size()
// Caller must make sure from>=0
inline TensorShapeProto::Dimension
multiplyDims(const TensorShapeProto& shape, int from, int upto_exclusive) {
  TensorShapeProto::Dimension dim;
  dim.set_dim_value(1);
  for (int i = from; i < upto_exclusive; ++i) {
    dim = dim * shape.dim(i);
  }
  return dim;
}

// propagate the element type from an input type to an output type.
// if an existing output element type exists, validate it matches.
inline void propagateElemTypeWithValidation(
    const TypeProto* input_type,
    TypeProto* output_type) {
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  if (input_type->value_case() != TypeProto::kTensorType) {
    fail_type_inference(
        "Input was expected to have tensor type. Got ",
        input_type->value_case());
  }

  if (input_type->tensor_type().elem_type() == TensorProto::UNDEFINED) {
    fail_type_inference("Element type of input was unknown");
  }

  if (output_type->value_case() == TypeProto::VALUE_NOT_SET) {
    output_type->mutable_tensor_type()->set_elem_type(
        input_type->tensor_type().elem_type());
  } else if (output_type->value_case() == TypeProto::kTensorType) {
    if (output_type->tensor_type().has_elem_type()) {
      if (input_type->tensor_type().elem_type() !=
          output_type->tensor_type().elem_type()) {
        fail_type_inference(
            "Input element type of ",
            input_type->tensor_type().elem_type(),
            " does not match existing output type of ",
            output_type->tensor_type().elem_type());
      }
    } else {
      output_type->mutable_tensor_type()->set_elem_type(
          input_type->tensor_type().elem_type());
    }
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output was expected to have tensor type. Got ",
        output_type->value_case());
  }
}

// Note: for all methods below for propagating type or shape, callers are
// responsible to handle optional inputs/outputs and ensure that the specified
// index value is less than NumInputs/NumOutputs.

inline void propagateElemTypeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type ||
      input_type->value_case() != TypeProto::kTensorType) {
    fail_type_inference("Input ", inputIndex, " expected to have tensor type");
  }
  if (input_type->tensor_type().elem_type() == TensorProto::UNDEFINED) {
    fail_type_inference("Element type of input ", inputIndex, " unknown");
  }
  auto output_type = ctx.getOutputType(outputIndex);
  if (output_type->value_case() == TypeProto::kTensorType ||
      output_type->value_case() == TypeProto::VALUE_NOT_SET) {
    output_type->mutable_tensor_type()->set_elem_type(
        input_type->tensor_type().elem_type());
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor type");
  }
}

inline void propagateElemTypeFromDtypeToOutput(
    InferenceContext& ctx,
    const int& data_type,
    size_t outputIndex) {
  auto attribute_tensor_datatype = data_type;
  auto output_type = ctx.getOutputType(outputIndex);
  if (output_type->value_case() == TypeProto::kTensorType ||
      output_type->value_case() == TypeProto::VALUE_NOT_SET) {
    output_type->mutable_tensor_type()->set_elem_type(
        attribute_tensor_datatype);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor type");
  }
}

inline void propagateElemTypeFromDtypeToOutput(
    InferenceContext& ctx,
    const AttributeProto* attr,
    size_t outputIndex) {
  if (attr->type() != AttributeProto::TENSOR) {
    fail_type_inference("Attribute expected to have tensor type");
  }
  if (attr->t().dims().size() != 1) {
    fail_type_inference("Attribute expected to have a one-dim tensor");
  }
  auto attribute_tensor_datatype = attr->t().data_type();
  propagateElemTypeFromDtypeToOutput(
      ctx, attribute_tensor_datatype, outputIndex);
}

inline bool hasShape(const TypeProto& type) {
  if (type.has_tensor_type()) {
    return type.tensor_type().has_shape();
  } else if (type.has_sequence_type() && type.sequence_type().has_elem_type()) {
    return hasShape(type.sequence_type().elem_type());
  }
  return false;
}

inline bool hasInputShape(InferenceContext& ctx, size_t n) {
  return ctx.getNumInputs() > static_cast<size_t>(n) && ctx.getInputType(n) &&
      hasShape(*ctx.getInputType(n));
}

inline bool hasNInputShapes(InferenceContext& ctx, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!hasInputShape(ctx, i)) {
      return false;
    }
  }
  return true;
}

inline const TensorShapeProto& getInputShape(InferenceContext& ctx, size_t n) {
  return ctx.getInputType(n)->tensor_type().shape();
}

// Caller must make sure fromDimIndex is strictly less than shape.dim_size()
inline void appendSingleDimCopiedFromInputTypeToOutputType(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex,
    size_t fromDimIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != output_type->value_case()) {
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor type");
  }
  if (TypeProto::kTensorType != input_type->value_case()) {
    fail_type_inference("Input ", inputIndex, " expected to have tensor type");
  }
  auto* dim = ctx.getOutputType(outputIndex)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->add_dim();
  *dim = input_type->tensor_type().shape().dim(static_cast<int>(fromDimIndex));
}

inline void propagateShapeFromInputToOutput(
    InferenceContext& ctx,
    size_t inputIndex,
    size_t outputIndex) {
  auto output_type = ctx.getOutputType(outputIndex);
  auto input_type = ctx.getInputType(inputIndex);
  if (TypeProto::kTensorType != input_type->value_case() ||
      TypeProto::kTensorType != output_type->value_case()) {
    throw std::runtime_error(ONNX_NAMESPACE::to_string(
        ctx.getInputType(inputIndex)->tensor_type().shape().dim_size()));
  }

  *ctx.getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape() =
      ctx.getInputType(inputIndex)->tensor_type().shape();
}

inline void propagateShapeAndTypeFromFirstInput(InferenceContext& ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  propagateShapeFromInputToOutput(ctx, 0, 0);
}

inline void updateOutputElemType(
    InferenceContext& ctx,
    size_t outputIndex,
    int32_t elemType) {
  auto output_type = ctx.getOutputType(outputIndex);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    output_type->mutable_tensor_type()->set_elem_type(elemType);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor type");
  }
}

// Infer type of an output from the value of a specified attribute, which is
// expected to have a valid value representing a TensorProto_DataType.
inline void propagateElemTypeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex,
    TensorProto_DataType default_value = TensorProto::UNDEFINED) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if (nullptr == attr_proto) { // attribute not present
    if (default_value != TensorProto::UNDEFINED) {
      updateOutputElemType(ctx, outputIndex, default_value);
      return;
    } else
      fail_type_inference(
          "Value of attribute ", attributeName, " not specified");
  }
  if (!attr_proto->has_i()) {
    fail_type_inference(
        "Attribute ",
        attributeName,
        " should be of integer type and specify a type.");
  }
  auto attr_value = attr_proto->i();
  auto elem_type = static_cast<TensorProto_DataType>(attr_value);
  if (!TensorProto_DataType_IsValid(elem_type)) {
    fail_type_inference(
        "Attribute ", attributeName, " does not specify a valid type.");
  }
  updateOutputElemType(ctx, outputIndex, elem_type);
}

inline TensorShapeProto* getOutputShape(InferenceContext& ctx, size_t n) {
  auto output_type = ctx.getOutputType(n);
  if ((output_type != nullptr) &&
      (output_type->value_case() == TypeProto::kTensorType ||
       output_type->value_case() == TypeProto::VALUE_NOT_SET)) {
    return output_type->mutable_tensor_type()->mutable_shape();
  } else
    fail_type_inference("Output ", n, " expected to have tensor type");
}

inline void appendDim(TensorShapeProto* shape, int64_t dim_value) {
  shape->add_dim()->set_dim_value(dim_value);
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorShapeProto& shape) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  *output_shape = shape;
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    const TensorProto& tensorProto) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  for (auto d : tensorProto.dims()) {
    auto* dim = output_shape->add_dim();
    dim->set_dim_value(d);
  }
}

inline void updateOutputShape(
    InferenceContext& ctx,
    size_t outputIndex,
    std::initializer_list<TensorShapeProto::Dimension> dims) {
  auto* output_shape = getOutputShape(ctx, outputIndex);
  for (auto& d : dims) {
    auto* dim = output_shape->add_dim();
    *dim = d;
  }
}

// Infer shape of an output from the value of a specified attribute, which is
// expected to be a list of integers specifying a valid shape.
inline void propagateShapeFromAttributeToOutput(
    InferenceContext& ctx,
    const std::string& attributeName,
    size_t outputIndex) {
  auto attr_proto = ctx.getAttribute(attributeName);
  if ((nullptr == attr_proto) || (!attr_proto->has_type()) ||
      (attr_proto->type() != AttributeProto_AttributeType_INTS)) {
    fail_shape_inference(
        "Attribute ", attributeName, " should specify a shape");
  }
  auto& int_list = attr_proto->ints();
  TensorShapeProto shape;
  for (auto dim_size : int_list) {
    if (dim_size < 0) {
      fail_shape_inference(
          "Negative values are not allowed in a shape specification");
    }
    shape.add_dim()->set_dim_value(dim_size);
  }

  updateOutputShape(ctx, outputIndex, shape);
}

inline void multidirectionalBroadcastShapeInference(
    const std::vector<const TensorShapeProto*>& shapes,
    TensorShapeProto& resultShape) {
  int result_shape_size = 0;
  // Get the result shape size.
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i]->dim_size() > result_shape_size) {
      result_shape_size = shapes[i]->dim_size();
    }
  }

  for (int i = 0; i < result_shape_size; ++i) {
    int64_t dim_value = 1;
    TensorShapeProto_Dimension symbolic_dim;
    int num_symbolic_dims = 0;
    for (size_t j = 0; j < shapes.size(); ++j) {
      if (i < result_shape_size - shapes[j]->dim_size()) {
        // Shape j will be filled with 1 at dimension i;
        continue;
      }

      auto dim_i_j =
          shapes[j]->dim(i - result_shape_size + shapes[j]->dim_size());
      if (dim_i_j.has_dim_value()) {
        if (dim_i_j.dim_value() != 1) {
          if (dim_value != dim_i_j.dim_value() && dim_value != 1) {
            fail_shape_inference("Incompatible dimensions");
          } else {
            dim_value = dim_i_j.dim_value();
          }
        }
      } else {
        if (num_symbolic_dims == 0) {
          symbolic_dim = dim_i_j;
          ++num_symbolic_dims;
        } else if (dim_i_j.dim_param() != symbolic_dim.dim_param()) {
          ++num_symbolic_dims;
        }
      }
    }

    if (dim_value != 1 || num_symbolic_dims == 0) {
      resultShape.add_dim()->set_dim_value(dim_value);
    } else if (num_symbolic_dims == 1) {
      *resultShape.add_dim() = symbolic_dim;
    } else {
      resultShape.add_dim();
    }
  }
}

inline void bidirectionalBroadcastShapeInference(
    const TensorShapeProto& shapeL,
    const TensorShapeProto& shapeR,
    TensorShapeProto& resultShape) {
  std::vector<const TensorShapeProto*> shapes;
  shapes.push_back(&shapeL);
  shapes.push_back(&shapeR);
  multidirectionalBroadcastShapeInference(shapes, resultShape);
}

/*
Merge the dimension information from two TensorShapeProto_Dimension instances.
Values are merged into target from source.
If target has no dimension information, copy from source.
If source has no dimension information, ignore source.
If both have dimension information:
 - Prefer values over params. If both have values, values must match.
 - Prefer target param over source param if mismatched.
Fail if there are mismatches in dimension values.
Currently, there is no way to refine/update dimension information for the
source from information available in the target.
*/
inline void mergeInDimensionInfo(
    const TensorShapeProto_Dimension& source_dim,
    TensorShapeProto_Dimension& target_dim,
    int dim_index) {
  // if source has value, merge into target
  // else if target has value, preserve it
  // else merge params
  if (source_dim.has_dim_value()) {
    auto source_value = source_dim.dim_value();
    if (target_dim.has_dim_value()) {
      auto target_value = target_dim.dim_value();
      if (target_value != source_value) {
        fail_shape_inference(
            "Can't merge shape info. "
            "Both source and target dimension have values but they differ. Source=",
            source_value,
            " Target=",
            target_value,
            " Dimension=",
            dim_index);
      }
    } else {
      target_dim.set_dim_value(source_value);
    }
  } else if (target_dim.has_dim_value()) {
    // if target has a value we preserve it so do nothing
  } else if (target_dim.has_dim_param()) {
    // prefer target param over source
  } else if (source_dim.has_dim_param()) {
    target_dim.set_dim_param(source_dim.dim_param());
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
inline void mergeInShapeInfo(
    const TensorShapeProto& source,
    TensorShapeProto& target) {
  auto num_source_dims = source.dim_size();
  auto num_target_dims = target.dim_size();
  if (num_source_dims != num_target_dims) {
    fail_shape_inference(
        "Mismatch between number of source and target dimensions. Source=",
        num_source_dims,
        " Target=",
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

inline void mergeInShapeInfo(
    const TensorShapeProto& source_shape,
    TypeProto_Tensor& target_type) {
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
inline void mergeInShapeInfo(
    const TypeProto_Tensor& source,
    TypeProto_Tensor& target) {
  if (source.has_shape())
    mergeInShapeInfo(source.shape(), target);
}

// Return a copy of a type, with a specified dimension removed from its shape.
inline TypeProto RemoveIthDimensionFromShape(
    const TypeProto& proto,
    int removed_dim) {
  TypeProto t(proto);
  auto mutable_shape = t.mutable_tensor_type()->mutable_shape();
  mutable_shape->clear_dim();

  const auto& dims = proto.tensor_type().shape().dim();

  for (int j = 0, end = dims.size(); j < end; ++j) {
    if (j != removed_dim)
      (*mutable_shape->add_dim()) = dims.Get(j);
  }

  return t;
}

// Return a copy of a type, with specified number of dimensions removed from the
// beginning.
inline TypeProto RemoveDimensionsFromShape(
    const TypeProto& proto,
    int num_dimensions) {
  TypeProto t(proto);
  auto mutable_shape = t.mutable_tensor_type()->mutable_shape();
  mutable_shape->clear_dim();

  const auto& dims = proto.tensor_type().shape().dim();

  // skip first num_dimensions
  for (int j = num_dimensions, end = dims.size(); j < end; ++j) {
    (*mutable_shape->add_dim()) = dims.Get(j);
  }

  return t;
}

// copied from GSL:
// https://github.com/Microsoft/GSL/blob/master/include/gsl/gsl_util
template <class T, class U>
static constexpr T narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

inline void
checkInputRank(InferenceContext& ctx, size_t input_index, int expected_rank) {
  // We check the rank only if a rank is known for the input:
  if (hasInputShape(ctx, input_index)) {
    auto rank = getInputShape(ctx, input_index).dim_size();
    if (rank != expected_rank)
      fail_shape_inference(
          "Input ",
          input_index,
          " expected to have rank ",
          expected_rank,
          " but has rank ",
          rank);
  }
}

// Unification (between dimensions and/or shapes) is at the heart of
// shape-inference. The current inference algorithm can check input
// shapes/dimensions of a node and update the output shapes/dimensions. It
// cannot currently update input shapes and dimensions (even though in some
// contexts this inference is possible). Hence, we have the variants below to
// support "const" and "mutable" dimensions/shapes in unification.

inline void checkDimEquality(int64_t value1, int64_t value2) {
  if (value1 != value2)
    fail_shape_inference(
        "Dimension mismatch in unification between ", value1, " and ", value2);
}

inline void unifyDim(const Dim& dim1, const Dim& dim2) {
  if (dim1.has_dim_value() && dim2.has_dim_value())
    checkDimEquality(dim1.dim_value(), dim2.dim_value());
}

// TODO: The functionality of unifyDim is similar to that of
// mergeInDimensionInfo. However, the error messages are different. Leaving this
// duplication in-place to preserve error message content.
inline void unifyDim(const Dim& source_dim, Dim& target_dim) {
  if (source_dim.has_dim_value()) {
    auto source_value = source_dim.dim_value();
    if (target_dim.has_dim_value()) {
      auto target_value = target_dim.dim_value();
      checkDimEquality(source_value, target_value);
    } else {
      target_dim.set_dim_value(source_value);
    }
  } else if (target_dim.has_dim_value()) {
    // if target has a value we preserve it.
    // we cannot set source dim value.
  } else if (target_dim.has_dim_param()) {
    // prefer target param over source
    // we cannot currently unify the dim_params
  } else if (source_dim.has_dim_param()) {
    target_dim.set_dim_param(source_dim.dim_param());
  }
}

inline void unifyInputDim(
    InferenceContext& ctx,
    size_t input_index,
    int dim_index,
    Dim& dim) {
  // We unify the dimensions only if it is available for specified input:
  if (hasInputShape(ctx, input_index)) {
    auto& input_shape = getInputShape(ctx, input_index);
    // This shape is expected to have rank > dim_index:
    if (input_shape.dim_size() <= dim_index)
      fail_shape_inference(
          "Input ",
          input_index,
          " expected to have rank >",
          dim_index,
          " but has rank ",
          input_shape.dim_size());
    const Dim& input_dim = input_shape.dim(dim_index);
    // Now, unify dim and input_dim:
    unifyDim(input_dim, dim);
  }
}

// unifyDim: unifies a dimension with a constant value. If the dimension
// already has a value, we check for equality of new value with old value.
inline void unifyDim(Dim& dim, int64_t value) {
  if (dim.has_dim_value()) {
    checkDimEquality(dim.dim_value(), value);
  } else
    dim.set_dim_value(value);
}

// target-shape = Union (target-shape, source_shape)
// Example 1: same rank, different dimensions
//    input1 shape: (2, 3, 4, 'x')
//    input2 shape: (2, 'y', 5, 'x')
//    output shape: (2, None, None, 'x')
// Example 2: different rank
//    input1 shape: (2, 3, 4, 'x')
//    input2 shape: (2, 3, 4)
//    output shape: None
inline void UnionShapeInfo(
    const TensorShapeProto& source_shape,
    TypeProto_Tensor& target_type) {
  if (target_type.has_shape()) {
    TensorShapeProto* target_shape = target_type.mutable_shape();

    auto source_rank = source_shape.dim_size();
    auto target_rank = target_shape->dim_size();
    if (source_rank != target_rank) {
      target_type.clear_shape();
      return;
    }

    for (int i = 0; i < source_rank; ++i) {
      const auto source_dim = source_shape.dim(i);
      const auto target_dim = target_shape->dim(i);
      bool is_dims_conflict = [&](){
        if (source_dim.has_dim_value()) {
          if (target_dim.has_dim_value() &&
              target_dim.dim_value() == source_dim.dim_value()) {
            return false;
          }
          return true;
        }

        if (source_dim.has_dim_param()) {
          if (target_dim.has_dim_param() &&
              target_dim.dim_param() == source_dim.dim_param()) {
            return false;
          }
          return true;
        }

        return (target_dim.has_dim_value() || target_dim.has_dim_param());
      }();
      if (is_dims_conflict &&
          (target_dim.has_dim_value() || target_dim.has_dim_param())) {
        auto dim = target_shape->mutable_dim(i);
        dim->clear_dim_value();
        dim->clear_dim_param();
      }
    }
  }
}

} // namespace ONNX_NAMESPACE
