#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>

#include "data_type_utils.h"

namespace onnx {
namespace Utils {

std::unordered_map<std::string, TypeProto>&
DataTypeUtils::GetTypeStrToProtoMap() {
  static std::unordered_map<std::string, TypeProto> map;
  return map;
}

void DataTypeUtils::Register(DataType type_key, const TypeProto& type_proto) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (GetTypeStrToProtoMap().find(*type_key) == GetTypeStrToProtoMap().end()) {
    GetTypeStrToProtoMap()[*type_key] = type_proto;
  } else {
    // One type is prevented from being registered multiple times
    // from different domain intentionally.
    assert(false);
  }
}

DataType DataTypeUtils::ToType(const std::string& type_str) {
  auto it = GetTypeStrToProtoMap().find(type_str);
  assert(it != GetTypeStrToProtoMap().end());
  return &(it->first);
}

DataType DataTypeUtils::ToType(const TypeProto& type_proto) {
  auto type_str = ToString(type_proto);
  return ToType(type_str);
}

const TypeProto& DataTypeUtils::ToTypeProto(const DataType& data_type) {
  auto it = GetTypeStrToProtoMap().find(*data_type);
  assert(it != GetTypeStrToProtoMap().end());
  return it->second;
}

std::string DataTypeUtils::ToString(const TypeProto& type_proto) {
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
DataType Abstract<T>::Type(const std::string& domain) {
  static Abstract abs_type(domain);
  return abs_type.TypeInternal();
}

template <int elemT>
DataType TensorType<elemT>::Type() {
  static TensorType tensor_type;
  return tensor_type.TypeInternal();
}

template DataType TensorType<TensorProto_DataType_FLOAT>::Type();
template DataType TensorType<TensorProto_DataType_UINT8>::Type();
template DataType TensorType<TensorProto_DataType_INT8>::Type();
template DataType TensorType<TensorProto_DataType_UINT16>::Type();
template DataType TensorType<TensorProto_DataType_INT16>::Type();
template DataType TensorType<TensorProto_DataType_INT32>::Type();
template DataType TensorType<TensorProto_DataType_INT64>::Type();
template DataType TensorType<TensorProto_DataType_STRING>::Type();
template DataType TensorType<TensorProto_DataType_BOOL>::Type();
template DataType TensorType<TensorProto_DataType_FLOAT16>::Type();
template DataType TensorType<TensorProto_DataType_DOUBLE>::Type();
template DataType TensorType<TensorProto_DataType_UINT32>::Type();
template DataType TensorType<TensorProto_DataType_UINT64>::Type();
template DataType TensorType<TensorProto_DataType_COMPLEX64>::Type();
template DataType TensorType<TensorProto_DataType_COMPLEX128>::Type();

template DataType Abstract<std::map<int64_t, std::string>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<int64_t, float>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<std::string, int64_t>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<int64_t, std::string>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<int64_t, float>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<int64_t, double>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<std::string, float>>::Type(
    const std::string& domain);
template DataType Abstract<std::map<std::string, double>>::Type(
    const std::string& domain);

} // namespace onnx
