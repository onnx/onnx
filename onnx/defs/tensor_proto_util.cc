// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/tensor_proto_util.h"

#include <string>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

int64_t ElementBitWidth(int32_t data_type) {
  switch (data_type) {
    case TensorProto::COMPLEX128:
      return 128;
    case TensorProto::DOUBLE:
    case TensorProto::COMPLEX64:
    case TensorProto::INT64:
    case TensorProto::UINT64:
      return 64;
    case TensorProto::FLOAT:
    case TensorProto::INT32:
    case TensorProto::UINT32:
      return 32;
    case TensorProto::INT16:
    case TensorProto::UINT16:
    case TensorProto::FLOAT16:
    case TensorProto::BFLOAT16:
      return 16;
    case TensorProto::INT8:
    case TensorProto::UINT8:
    case TensorProto::BOOL:
    case TensorProto::FLOAT8E4M3FN:
    case TensorProto::FLOAT8E4M3FNUZ:
    case TensorProto::FLOAT8E5M2:
    case TensorProto::FLOAT8E5M2FNUZ:
    case TensorProto::FLOAT8E8M0:
      return 8;
    case TensorProto::UINT4:
    case TensorProto::INT4:
    case TensorProto::FLOAT4E2M1:
      return 4;
    case TensorProto::UINT2:
    case TensorProto::INT2:
      return 2;
    default:
      return -1;
  }
}

int64_t RawDataElementCount(const TensorProto& tensor, size_t element_size, bool exact_fit) {
  const int64_t num_elements = safe_dim_product(
      tensor.dims(), [&](const char* msg) { fail_shape_inference(msg, " for tensor: ", tensor.name()); });
  const size_t size = tensor.raw_data().size();
  // Divide rather than multiply so no product can overflow.
  const auto available = static_cast<uint64_t>(size / element_size);
  const auto required = static_cast<uint64_t>(num_elements);
  if (exact_fit ? (available != required || size % element_size != 0) : available < required) {
    fail_shape_inference(
        "Data size mismatch. Tensor: ",
        tensor.name(),
        " has ",
        size,
        " bytes of raw_data for ",
        num_elements,
        " elements of size ",
        element_size);
  }
  return num_elements;
}

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
    for (const auto& val : values) {                            \
      t.add_##field##_data(val);                                \
    }                                                           \
    return t;                                                   \
  }

#define DEFINE_PARSE_DATA(type, typed_data_fetch, tensorproto_datatype)                                            \
  template <>                                                                                                      \
  std::vector<type> ParseData(const TensorProto* tensor_proto) {                                                   \
    if (!tensor_proto->has_data_type() || tensor_proto->data_type() == TensorProto_DataType_UNDEFINED) {           \
      fail_shape_inference("The type of tensor: ", tensor_proto->name(), " is undefined so it cannot be parsed."); \
    } else if (tensor_proto->data_type() != (tensorproto_datatype)) {                                              \
      fail_shape_inference(                                                                                        \
          "ParseData type mismatch for tensor: ",                                                                  \
          tensor_proto->name(),                                                                                    \
          ". Expected:",                                                                                           \
          Utils::DataTypeUtils::ToDataTypeString(tensorproto_datatype),                                            \
          " Actual:",                                                                                              \
          Utils::DataTypeUtils::ToDataTypeString(tensor_proto->data_type()));                                      \
    }                                                                                                              \
    if (tensor_proto->has_data_location() && tensor_proto->data_location() == TensorProto_DataLocation_EXTERNAL) { \
      fail_shape_inference(                                                                                        \
          "Cannot parse data from external tensors. Please ",                                                      \
          "load external data into raw data for tensor: ",                                                         \
          tensor_proto->name());                                                                                   \
    } else if (!tensor_proto->has_raw_data()) {                                                                    \
      const int64_t num_elements = safe_dim_product(tensor_proto->dims(), [&](const char* msg) {                   \
        fail_shape_inference(msg, " for tensor: ", tensor_proto->name());                                          \
      });                                                                                                          \
      std::vector<type> res;                                                                                       \
      const auto& data = tensor_proto->typed_data_fetch();                                                         \
      if (data.size() != num_elements) {                                                                           \
        fail_shape_inference(                                                                                      \
            "Data size mismatch. Tensor: ",                                                                        \
            tensor_proto->name(),                                                                                  \
            " expected num elements ",                                                                             \
            num_elements,                                                                                          \
            " does not match the actual num elements ",                                                            \
            data.size());                                                                                          \
      }                                                                                                            \
      res.insert(res.end(), data.begin(), data.end());                                                             \
      return res;                                                                                                  \
    }                                                                                                              \
    if (tensor_proto->data_type() == TensorProto_DataType_STRING) {                                                \
      fail_shape_inference(                                                                                        \
          tensor_proto->name(),                                                                                    \
          " data type is string. string",                                                                          \
          " content is required to be stored in repeated bytes string_data field.",                                \
          " raw_data type cannot be string.");                                                                     \
    }                                                                                                              \
    return ParseRawData<type>(*tensor_proto);                                                                      \
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

DEFINE_PARSE_DATA(int32_t, int32_data, TensorProto_DataType_INT32)
DEFINE_PARSE_DATA(int64_t, int64_data, TensorProto_DataType_INT64)
DEFINE_PARSE_DATA(float, float_data, TensorProto_DataType_FLOAT)
DEFINE_PARSE_DATA(double, double_data, TensorProto_DataType_DOUBLE)

#undef DEFINE_PARSE_DATA

} // namespace ONNX_NAMESPACE
