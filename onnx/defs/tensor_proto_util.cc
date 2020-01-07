// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "tensor_proto_util.h"
#include <vector>
#include "onnx/common/platform_helpers.h"

namespace ONNX_NAMESPACE {

#define DEFINE_TO_TENSOR_ONE(type, enumType, field) \
  template <>                                       \
  TensorProto ToTensor<type>(const type& value) {   \
    TensorProto t;                                  \
    t.set_data_type(enumType);                      \
    t.add_##field##_data(value);                    \
    return t;                                       \
  }

#define DEFINE_TO_TENSOR_LIST(type, enumType, field)            \
  template <>                                                   \
  TensorProto ToTensor<type>(const std::vector<type>& values) { \
    TensorProto t;                                              \
    t.clear_##field##_data();                                   \
    t.set_data_type(enumType);                                  \
    for (const type& val : values) {                            \
      t.add_##field##_data(val);                                \
    }                                                           \
    return t;                                                   \
  }

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                          \
  template <>                                                              \
  const std::vector<type> ParseData(const TensorProto* tensor_proto) {     \
    std::vector<type> res;                                                 \
    if (!tensor_proto->has_raw_data()) {                                   \
      const auto& data = tensor_proto->typed_data_fetch();                 \
      res.insert(res.end(), data.begin(), data.end());                     \
      return res;                                                          \
    }                                                                      \
    /* make copy as we may have to reverse bytes */                        \
    std::string raw_data = tensor_proto->raw_data();                       \
    /* okay to remove const qualifier as we have already made a copy */    \
    char* bytes = const_cast<char*>(raw_data.c_str());                     \
    /*onnx is little endian serialized always-tweak byte order if needed*/ \
    if (!is_processor_little_endian()) {                                   \
      const size_t element_size = sizeof(type);                            \
      const size_t num_elements = raw_data.size() / element_size;          \
      for (size_t i = 0; i < num_elements; ++i) {                          \
        char* start_byte = bytes + i * element_size;                       \
        char* end_byte = start_byte + element_size - 1;                    \
        /* keep swapping */                                                \
        for (size_t count = 0; count < element_size / 2; ++count) {        \
          char temp = *start_byte;                                         \
          *start_byte = *end_byte;                                         \
          *end_byte = temp;                                                \
          ++start_byte;                                                    \
          --end_byte;                                                      \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    res.insert(                                                            \
        res.end(),                                                         \
        reinterpret_cast<const type*>(bytes),                              \
        reinterpret_cast<const type*>(bytes + raw_data.size()));           \
    return res;                                                            \
  }

DEFINE_TO_TENSOR_ONE(float, TensorProto_DataType_FLOAT, float)
DEFINE_TO_TENSOR_ONE(bool, TensorProto_DataType_BOOL, int32)
DEFINE_TO_TENSOR_ONE(int32_t, TensorProto_DataType_INT32, int32)
DEFINE_TO_TENSOR_ONE(int64_t, TensorProto_DataType_INT64, int64)
DEFINE_TO_TENSOR_ONE(uint64_t, TensorProto_DataType_UINT64, uint64)
DEFINE_TO_TENSOR_ONE(double, TensorProto_DataType_DOUBLE, double)
DEFINE_TO_TENSOR_ONE(std::string, TensorProto_DataType_STRING, string)

DEFINE_TO_TENSOR_LIST(float, TensorProto_DataType_FLOAT, float)
DEFINE_TO_TENSOR_LIST(bool, TensorProto_DataType_BOOL, int32)
DEFINE_TO_TENSOR_LIST(int32_t, TensorProto_DataType_INT32, int32)
DEFINE_TO_TENSOR_LIST(int64_t, TensorProto_DataType_INT64, int64)
DEFINE_TO_TENSOR_LIST(uint64_t, TensorProto_DataType_UINT64, uint64)
DEFINE_TO_TENSOR_LIST(double, TensorProto_DataType_DOUBLE, double)
DEFINE_TO_TENSOR_LIST(std::string, TensorProto_DataType_STRING, string)

DEFINE_PARSE_DATA(int32_t, int32_data)
DEFINE_PARSE_DATA(int64_t, int64_data)
DEFINE_PARSE_DATA(float, float_data)
DEFINE_PARSE_DATA(double, double_data)

} // namespace ONNX_NAMESPACE
