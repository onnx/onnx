// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Internal helpers shared between ir_pb_converter.cc and other onnx-internal
// translation units that need small, single-value conversions between the
// protobuf types and ir.h's Graph/Node/Value/Tensor -- as opposed to
// ir_pb_converter.h's public, whole-Graph/whole-ModelProto entry points. Not
// part of onnx's public API. Added for Graph-native shape inference
// (onnx/common/graph_shape_inference.h), which builds a single node's
// NodeProto and a single value's TypeProto/TensorProto on the fly, without
// converting the whole surrounding graph.

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

// Converts a TensorShapeProto to ir.h's Dimension vector (unknown dims become
// a default-constructed Dimension). Defined in ir_pb_converter.cc.
std::vector<Dimension> tensorShapeProtoToDimensions(const TensorShapeProto& tsp);

// Encodes a Value's elemType()/sizes() into a TypeProto_Tensor. Defined in
// ir_pb_converter.cc.
void encodeTypeProtoTensorType(TypeProto_Tensor& tensor_type, const Value& n);

// Encodes one attribute of Node `n` (by name) into NodeProto `n_p`. Defined
// in ir_pb_converter.cc.
void addAttribute(NodeProto& n_p, const Node& n, Symbol name);

// Encodes a Tensor's contents into a TensorProto (always copying -- callers
// build small, ephemeral TensorProtos, not a ModelProto export). Defined in
// ir_pb_converter.cc.
void encodeTensor(TensorProto& p, const Tensor& tensor);

} // namespace ONNX_NAMESPACE
