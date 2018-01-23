#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>

#include "data_type_utils.h"

namespace onnx {
namespace Utils {

std::unordered_map<std::string, PDataType>&
DataTypeUtils::GetTypeIdToDataMap() {
  static std::unordered_map<std::string, PDataType> map;
  return map;
}

void DataTypeUtils::Register(PDataType p_data_type) {
  assert(nullptr != p_data_type);
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (GetTypeIdToDataMap().find(p_data_type->Id()) ==
      GetTypeIdToDataMap().end()) {
    GetTypeIdToDataMap()[p_data_type->Id()] = p_data_type;
  } else {
    // One type is prevented from being registered multiple times
    // from different domain intentionally.
    assert(false);
  }
}

PDataType DataTypeUtils::ToType(const std::string& type_id) {
  auto it = GetTypeIdToDataMap().find(type_id);
  assert(it != GetTypeIdToDataMap().end());
  return it->second;
}

PDataType DataTypeUtils::ToType(const TypeProto& type_proto) {
  auto type_id = GetId(type_proto);
  return ToType(type_id);
}

std::string DataTypeUtils::GetId(const TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      // Tensor type.
      return "tensor(" +
          DataTypeUtils::GetElementTypeStr(
                 type_proto.tensor_type().elem_type()) +
          ")";
    }
    case TypeProto::ValueCase::kAbsType: {
      // Abstract type.
      return type_proto.abs_type().identifier();
    }
    default:
      assert(false);
      return "";
  }
}

std::string DataTypeUtils::GetElementTypeStr(TensorProto_DataType elem_type) {
  switch (elem_type) {
    case onnx::TensorProto_DataType_UNDEFINED:
      return "undefined";
    case onnx::TensorProto_DataType_FLOAT:
      return "float";
    case onnx::TensorProto_DataType_UINT8:
      return "uint8";
    case onnx::TensorProto_DataType_INT8:
      return "int8";
    case onnx::TensorProto_DataType_UINT16:
      return "uint16";
    case onnx::TensorProto_DataType_INT16:
      return "int16";
    case onnx::TensorProto_DataType_INT32:
      return "int32";
    case onnx::TensorProto_DataType_INT64:
      return "int64";
    case onnx::TensorProto_DataType_STRING:
      return "string";
    case onnx::TensorProto_DataType_BOOL:
      return "bool";
    case onnx::TensorProto_DataType_FLOAT16:
      return "float16";
    case onnx::TensorProto_DataType_DOUBLE:
      return "double";
    case onnx::TensorProto_DataType_UINT32:
      return "uint32";
    case onnx::TensorProto_DataType_UINT64:
      return "uint64";
    case onnx::TensorProto_DataType_COMPLEX64:
      return "complex64";
    case onnx::TensorProto_DataType_COMPLEX128:
      return "complex128";
    default:
      assert(false);
      return "";
  }
}
} // namespace Utils

template <typename T>
PDataType Abstract<T>::Type(
    const std::string& name,
    const std::string& domain) {
  static Abstract abs_type(name, domain);
  return &abs_type;
}

template <int elemT>
PDataType TensorType<elemT>::Type() {
  static TensorType tensor_type;
  return &tensor_type;
}

const std::string& DataType::Domain() const {
  return domain;
}
const std::string& DataType::Id() const {
  return id;
}
const std::string& DataType::Description() const {
  return description;
}
const TypeProto& DataType::ToProto() const {
  return type_proto;
}

PDataType DataType::Tensor_FLOAT =
    TensorType<TensorProto_DataType_FLOAT>::Type();
PDataType DataType::Tensor_UINT8 =
    TensorType<TensorProto_DataType_UINT8>::Type();
PDataType DataType::Tensor_INT8 = TensorType<TensorProto_DataType_INT8>::Type();
PDataType DataType::Tensor_UINT16 =
    TensorType<TensorProto_DataType_UINT16>::Type();
PDataType DataType::Tensor_INT16 =
    TensorType<TensorProto_DataType_INT16>::Type();
PDataType DataType::Tensor_INT32 =
    TensorType<TensorProto_DataType_INT32>::Type();
PDataType DataType::Tensor_INT64 =
    TensorType<TensorProto_DataType_INT64>::Type();
PDataType DataType::Tensor_STRING =
    TensorType<TensorProto_DataType_STRING>::Type();
PDataType DataType::Tensor_BOOL = TensorType<TensorProto_DataType_BOOL>::Type();
PDataType DataType::Tensor_FLOAT16 =
    TensorType<TensorProto_DataType_FLOAT16>::Type();
PDataType DataType::Tensor_DOUBLE =
    TensorType<TensorProto_DataType_DOUBLE>::Type();
PDataType DataType::Tensor_UINT32 =
    TensorType<TensorProto_DataType_UINT32>::Type();
PDataType DataType::Tensor_UINT64 =
    TensorType<TensorProto_DataType_UINT64>::Type();
PDataType DataType::Tensor_COMPLEX64 =
    TensorType<TensorProto_DataType_COMPLEX64>::Type();
PDataType DataType::Tensor_COMPLEX128 =
    TensorType<TensorProto_DataType_COMPLEX128>::Type();

PDataType DataType::Map_Int64_String =
    Abstract<std::map<int64_t, std::string>>::Type("map(int64, string)");
PDataType DataType::Map_Int64_Float =
    Abstract<std::map<int64_t, float>>::Type("map(int64, float)");
PDataType DataType::Map_String_Int64 =
    Abstract<std::map<std::string, int64_t>>::Type("map(string, int64)");
PDataType DataType::Map_Int64_Double =
    Abstract<std::map<int64_t, double>>::Type("map(int64, double)");
PDataType DataType::Map_String_Float =
    Abstract<std::map<std::string, float>>::Type("map(string, float)");
PDataType DataType::Map_String_Double =
    Abstract<std::map<std::string, double>>::Type("map(string, double)");
} // namespace onnx
