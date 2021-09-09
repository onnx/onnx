/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shape_inference.h"

namespace ONNX_NAMESPACE {

/// <summary>
/// Utility function for UnionShapeInfoForTensor.
/// Both shapes must be of the same rank
/// </summary>
/// <param name="source_shape"></param>
/// <param name="target_shape">destination shape</param>
void UnionShapeInfo(const TensorShapeProto& source_shape, TensorShapeProto& target_shape) {
  auto source_rank = source_shape.dim_size();
  for (int i = 0; i < source_rank; ++i) {
    const auto source_dim = source_shape.dim(i);
    const auto target_dim = target_shape.dim(i);
    bool is_dims_conflict = [&]() {
      if (source_dim.has_dim_value()) {
        if (target_dim.has_dim_value() && target_dim.dim_value() == source_dim.dim_value()) {
          return false;
        }
        return true;
      }

      if (source_dim.has_dim_param()) {
        if (target_dim.has_dim_param() && target_dim.dim_param() == source_dim.dim_param()) {
          return false;
        }
        return true;
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

template<typename TENSOR_TYPE>
void UnionShapeInfoForTensor(const TensorShapeProto& source_shape, TENSOR_TYPE& target_type) {
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

void UnionShapeInfo(const TensorShapeProto& source_shape, TypeProto_SparseTensor& target_type) {
  UnionShapeInfoForTensor(source_shape, target_type);
}


void UnionTypeInfo(const TypeProto& source_type, TypeProto& target_type) {
  if (source_type.value_case() != target_type.value_case()) {
    fail_type_inference("Mismatched type:", " source=", source_type.value_case(), " target=", target_type.value_case());
  }

  const auto target_case = target_type.value_case();
  if (target_case == TypeProto::ValueCase::kTensorType) {
    auto source_elem_type = source_type.tensor_type().elem_type();
    auto target_elem_type = target_type.tensor_type().elem_type();

    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched tensor element type:", " source=", source_elem_type, " target=", target_elem_type);
    }

    UnionShapeInfoForTensor(source_type.tensor_type().shape(), *target_type.mutable_tensor_type());
  } else if (target_case == TypeProto::ValueCase::kSparseTensorType) {
    auto source_elem_type = source_type.sparse_tensor_type().elem_type();
    auto target_elem_type = target_type.sparse_tensor_type().elem_type();
    if (source_elem_type != target_elem_type) {
      fail_type_inference(
          "Mismatched sparse tensor element type:", " source=", source_elem_type, " target=", target_elem_type);
    }

    UnionShapeInfoForTensor(source_type.sparse_tensor_type().shape(), *target_type.mutable_sparse_tensor_type());
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
  }
}


} // namespace ONNX_NAMESPACE