// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "onnx/onnx2/cpu/onnx2.h"

namespace ONNX_NAMESPACE {
namespace v2 {

/**
 * The function propulates external data for every tensor.
 * The function does not remove anything from the model.
 */
void PopulateExternalData(ModelProto& model, size_t threshold, const std::string& external_data_location);

/**
 * Clears the external data from the model.
 */
void ClearExternalData(ModelProto& model);

/**
 * IteratorTensorProto is an iterator that traverses all TensorProto objects.
 */
class IteratorTensorProto {
 protected:
  struct Position {
    GraphProto* graph;
    int node_index = 0;
    int attr_index = 0;
    int node_initializer_index = 0;
  };

 public:
  explicit inline IteratorTensorProto(GraphProto* graph) : tp_(nullptr), positions_() {
    positions_.emplace_back(Position{graph});
  }
  inline TensorProto& operator*() {
    return *tp_;
  }
  inline TensorProto* operator->() {
    return tp_;
  }
  bool next();

 private:
  TensorProto* tp_;
  std::vector<Position> positions_;
};

//////////////
// Serializing
//////////////

/**
 * The function saves the ONNX model to a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 */
template <typename T>
inline void SerializeProtoToStream(T&, utils::BinaryWriteStream&, SerializeOptions&, bool clear_external_data = true) {
  EXT_THROW(
      "SerializeProtoToStream is not implemented for type ",
      typeid(T).name(),
      ", clear_external_data=",
      clear_external_data);
}

/**
 * The function saves the ONNX model to a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 */
void SerializeModelProtoToStream(
    ModelProto& model,
    utils::BinaryWriteStream& stream,
    SerializeOptions& options,
    bool clear_external_data = true);

template <>
inline void SerializeProtoToStream(
    ModelProto& model,
    utils::BinaryWriteStream& stream,
    SerializeOptions& options,
    bool clear_external_data) {
  SerializeModelProtoToStream(model, stream, options, clear_external_data);
}

//////////
// PArsing
//////////

/**
 * The function reads the ONNX model from a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 */
template <typename T>
inline void ParseProtoFromStream(T&, utils::BinaryStream&, ParseOptions&, bool clear_external_data = true) {
  EXT_THROW(
      "ParseProtoFromStream is not implemented for type ",
      typeid(T).name(),
      ", clear_external_data=",
      clear_external_data);
}

/**
 * The function saves the ONNX model to a binary stream.
 * If external weights is triggered, the model is modified to add external data.
 */
void ParseModelProtoFromStream(
    ModelProto& model,
    utils::BinaryStream& stream,
    ParseOptions& options,
    bool clear_external_data = true);

template <>
inline void
ParseProtoFromStream(ModelProto& model, utils::BinaryStream& stream, ParseOptions& options, bool clear_external_data) {
  ParseModelProtoFromStream(model, stream, options, clear_external_data);
}

} // namespace v2
} // namespace ONNX_NAMESPACE