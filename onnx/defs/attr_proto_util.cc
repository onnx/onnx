// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "attr_proto_util.h"

#include <string>
#include <utility>
#include <vector>

namespace ONNX_NAMESPACE {

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                  \
  AttributeProto MakeAttribute(std::string attr_name, type value) { \
    AttributeProto a;                                               \
    a.set_name(std::move(attr_name));                               \
    a.set_type(enumType);                                           \
    a.set_##field(value);                                           \
    return a;                                                       \
  }

#define ADD_ATTR_IMPL(type, enumType, field)                        \
  AttributeProto MakeAttribute(std::string attr_name, type value) { \
    AttributeProto a;                                               \
    a.set_name(std::move(attr_name));                               \
    a.set_type(enumType);                                           \
    *(a.mutable_##field()) = std::move(value);                      \
    return a;                                                       \
  }

#define ADD_LIST_ATTR_IMPL(type, enumType, field)                                 \
  AttributeProto MakeAttribute(std::string attr_name, std::vector<type> values) { \
    AttributeProto a;                                                             \
    a.set_name(std::move(attr_name));                                             \
    a.set_type(enumType);                                                         \
    for (auto&& val : std::move(values)) {                                        \
      *(a.mutable_##field()->Add()) = std::move(val);                             \
    }                                                                             \
    return a;                                                                     \
  }

ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType_FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType_INT, i)
ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType_STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType_TENSOR, t)
ADD_ATTR_IMPL(GraphProto, AttributeProto_AttributeType_GRAPH, g)
ADD_ATTR_IMPL(TypeProto, AttributeProto_AttributeType_TYPE_PROTO, tp)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType_FLOATS, floats)
// NOLINTNEXTLINE(performance-unnecessary-value-param)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType_INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType_STRINGS, strings)
ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType_TENSORS, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType_GRAPHS, graphs)
ADD_LIST_ATTR_IMPL(TypeProto, AttributeProto_AttributeType_TYPE_PROTOS, type_protos)

AttributeProto MakeRefAttribute(const std::string& attr_name, AttributeProto_AttributeType type) {
  return MakeRefAttribute(attr_name, attr_name, type);
}

AttributeProto MakeRefAttribute(
    const std::string& attr_name,
    const std::string& referred_attr_name,
    AttributeProto_AttributeType type) {
  AttributeProto a;
  a.set_name(attr_name);
  a.set_ref_attr_name(referred_attr_name);
  a.set_type(type);
  return a;
}

} // namespace ONNX_NAMESPACE
