// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/data_type_utils.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ONNX_NAMESPACE::Utils {
namespace {

// ASCII whitespace only; trimming must not depend on the locale.
std::string_view StripWhitespace(std::string_view str) {
  constexpr std::string_view whitespace = " \t\n\v\f\r";
  const auto first = str.find_first_not_of(whitespace);
  if (first == std::string_view::npos)
    return {};
  return str.substr(first, str.find_last_not_of(whitespace) - first + 1);
}

std::string_view StripParens(std::string_view str) {
  str = StripWhitespace(str);
  if (!str.empty() && str.front() == '(')
    str.remove_prefix(1);
  str = StripWhitespace(str);
  if (!str.empty() && str.back() == ')')
    str.remove_suffix(1);
  return StripWhitespace(str);
}

// Singleton wrapper around allowed data types.
// This implements construct on first use which is needed to ensure
// static objects are initialized before use. Ops registration does not work
// properly without this.
class TypesWrapper final {
 public:
  static TypesWrapper& GetTypesWrapper();

  std::unordered_set<std::string>& GetAllowedDataTypes();

  std::unordered_map<std::string, int32_t>& TypeStrToTensorDataType();

  std::unordered_map<int32_t, std::string>& TensorDataTypeToTypeStr();

  ~TypesWrapper() = default;
  TypesWrapper(const TypesWrapper&) = delete;
  void operator=(const TypesWrapper&) = delete;

 private:
  TypesWrapper();

  std::unordered_map<std::string, int> type_str_to_tensor_data_type_;
  std::unordered_map<int, std::string> tensor_data_type_to_type_str_;
  std::unordered_set<std::string> allowed_data_types_;
};

} // namespace

std::unordered_map<std::string, TypeProto>& DataTypeUtils::GetTypeStrToProtoMap() {
  static std::unordered_map<std::string, TypeProto> map;
  return map;
}

std::mutex& DataTypeUtils::GetTypeStrLock() {
  static std::mutex lock;
  return lock;
}

DataType DataTypeUtils::ToType(const TypeProto& type_proto) {
  auto typeStr = ToString(type_proto);
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(typeStr);
  if (it == GetTypeStrToProtoMap().end()) {
    TypeProto type;
    FromString(typeStr, type);
    GetTypeStrToProtoMap()[typeStr] = type;
    it = GetTypeStrToProtoMap().find(typeStr);
  }
  return &it->first;
}

DataType DataTypeUtils::ToType(const std::string& type_str) {
  TypeProto type;
  FromString(type_str, type);
  return ToType(type);
}

const TypeProto& DataTypeUtils::ToTypeProto(const DataType& data_type) {
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(*data_type);
  if (GetTypeStrToProtoMap().end() == it) {
    ONNX_THROW_EX(std::invalid_argument("Invalid data type " + *data_type));
  }
  return it->second;
}

std::string DataTypeUtils::ToString(const TypeProto& type_proto, const std::string& left, const std::string& right) {
  switch (type_proto.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      // Note: We do not distinguish tensors with zero rank (a shape consisting
      // of an empty sequence of dimensions) here.
      return left + "tensor(" + ToDataTypeString(type_proto.tensor_type().elem_type()) + ")" + right;
    }
    case TypeProto::ValueCase::kSequenceType: {
      return ToString(type_proto.sequence_type().elem_type(), left + "seq(", ")" + right);
    }
    case TypeProto::ValueCase::kOptionalType: {
      return ToString(type_proto.optional_type().elem_type(), left + "optional(", ")" + right);
    }
    case TypeProto::ValueCase::kMapType: {
      std::string map_str = "map(" + ToDataTypeString(type_proto.map_type().key_type()) + ",";
      return ToString(type_proto.map_type().value_type(), left + map_str, ")" + right);
    }
    case TypeProto::ValueCase::kOpaqueType: {
      std::string result;
      const auto& op_type = type_proto.opaque_type();
      result.append(left).append("opaque(");
      if (op_type.has_domain() && !op_type.domain().empty()) {
        result.append(op_type.domain()).append(",");
      }
      if (op_type.has_name() && !op_type.name().empty()) {
        result.append(op_type.name());
      }
      result.append(")").append(right);
      return result;
    }
    case TypeProto::ValueCase::kSparseTensorType: {
      // Note: We do not distinguish tensors with zero rank (a shape consisting
      // of an empty sequence of dimensions) here.
      return left + "sparse_tensor(" + ToDataTypeString(type_proto.sparse_tensor_type().elem_type()) + ")" + right;
    }
    default:
      ONNX_THROW_EX(
          std::invalid_argument("Unsupported type proto value case:" + std::to_string(type_proto.value_case())));
  }
}

std::string DataTypeUtils::ToDataTypeString(int32_t tensor_data_type) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  auto iter = t.TensorDataTypeToTypeStr().find(tensor_data_type);
  if (t.TensorDataTypeToTypeStr().end() == iter) {
    ONNX_THROW_EX(std::invalid_argument("Invalid tensor data type " + std::to_string(tensor_data_type) + "."));
  }
  return iter->second;
}

void DataTypeUtils::FromString(std::string_view type_str, TypeProto& type_proto) {
  auto s = StripWhitespace(type_str);
  type_proto.Clear();
  if (s.substr(0, 3) == "seq") {
    s = StripParens(s.substr(3));
    FromString(s, *type_proto.mutable_sequence_type()->mutable_elem_type());
    return;
  } else if (s.substr(0, 8) == "optional") {
    s = StripParens(s.substr(8));
    FromString(s, *type_proto.mutable_optional_type()->mutable_elem_type());
    return;
  } else if (s.substr(0, 3) == "map") {
    s = StripParens(s.substr(3));
    const auto key_size = s.find(',');
    const auto key = StripWhitespace(s.substr(0, key_size));
    if (key_size != std::string_view::npos)
      s.remove_prefix(key_size + 1);
    auto key_type = FromDataTypeString(std::string(key));
    type_proto.mutable_map_type()->set_key_type(key_type);
    FromString(s, *type_proto.mutable_map_type()->mutable_value_type());
    return;
  }
  if (s.substr(0, 6) == "opaque") {
    auto* opaque_type = type_proto.mutable_opaque_type();
    s = StripParens(s.substr(6));
    if (!s.empty()) {
      size_t cm = s.find(',');
      if (cm != std::string_view::npos) {
        if (cm > 0) {
          opaque_type->mutable_domain()->assign(s.substr(0, cm));
        }
        s.remove_prefix(cm + 1); // skip comma
      }
      if (!s.empty()) {
        opaque_type->mutable_name()->assign(s);
      }
    }
    return;
  }
  if (s.substr(0, 13) == "sparse_tensor") {
    s = StripParens(s.substr(13));
    auto e = FromDataTypeString(std::string(s));
    type_proto.mutable_sparse_tensor_type()->set_elem_type(e);
  } else if (s.substr(0, 6) == "tensor") {
    s = StripParens(s.substr(6));
    auto e = FromDataTypeString(std::string(s));
    type_proto.mutable_tensor_type()->set_elem_type(e);
  } else {
    // Scalar
    auto e = FromDataTypeString(std::string(s));
    TypeProto::Tensor* t = type_proto.mutable_tensor_type();
    t->set_elem_type(e);
    // Call mutable_shape() to initialize a shape with no dimension.
    t->mutable_shape();
  }
} // namespace Utils

bool DataTypeUtils::IsValidDataTypeString(const std::string& type_str) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  const auto& allowedSet = t.GetAllowedDataTypes();
  return (allowedSet.find(type_str) != allowedSet.end());
}

int32_t DataTypeUtils::FromDataTypeString(const std::string& type_str) {
  if (!IsValidDataTypeString(type_str)) {
    ONNX_THROW_EX(
        std::invalid_argument(
            "DataTypeUtils::FromDataTypeString - Received invalid data type string '" + type_str + "'."));
  }

  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  return t.TypeStrToTensorDataType()[type_str];
}

namespace {

TypesWrapper& TypesWrapper::GetTypesWrapper() {
  static TypesWrapper types;
  return types;
}

std::unordered_set<std::string>& TypesWrapper::GetAllowedDataTypes() {
  return allowed_data_types_;
}

std::unordered_map<std::string, int>& TypesWrapper::TypeStrToTensorDataType() {
  return type_str_to_tensor_data_type_;
}

std::unordered_map<int, std::string>& TypesWrapper::TensorDataTypeToTypeStr() {
  return tensor_data_type_to_type_str_;
}

TypesWrapper::TypesWrapper() {
  // DataType strings. These should match the DataTypes defined in onnx.proto
  type_str_to_tensor_data_type_["float"] = TensorProto_DataType_FLOAT;
  type_str_to_tensor_data_type_["float16"] = TensorProto_DataType_FLOAT16;
  type_str_to_tensor_data_type_["bfloat16"] = TensorProto_DataType_BFLOAT16;
  type_str_to_tensor_data_type_["double"] = TensorProto_DataType_DOUBLE;
  type_str_to_tensor_data_type_["int8"] = TensorProto_DataType_INT8;
  type_str_to_tensor_data_type_["int16"] = TensorProto_DataType_INT16;
  type_str_to_tensor_data_type_["int32"] = TensorProto_DataType_INT32;
  type_str_to_tensor_data_type_["int64"] = TensorProto_DataType_INT64;
  type_str_to_tensor_data_type_["uint8"] = TensorProto_DataType_UINT8;
  type_str_to_tensor_data_type_["uint16"] = TensorProto_DataType_UINT16;
  type_str_to_tensor_data_type_["uint32"] = TensorProto_DataType_UINT32;
  type_str_to_tensor_data_type_["uint64"] = TensorProto_DataType_UINT64;
  type_str_to_tensor_data_type_["complex64"] = TensorProto_DataType_COMPLEX64;
  type_str_to_tensor_data_type_["complex128"] = TensorProto_DataType_COMPLEX128;
  type_str_to_tensor_data_type_["string"] = TensorProto_DataType_STRING;
  type_str_to_tensor_data_type_["bool"] = TensorProto_DataType_BOOL;
  type_str_to_tensor_data_type_["float8e4m3fn"] = TensorProto_DataType_FLOAT8E4M3FN;
  type_str_to_tensor_data_type_["float8e4m3fnuz"] = TensorProto_DataType_FLOAT8E4M3FNUZ;
  type_str_to_tensor_data_type_["float8e5m2"] = TensorProto_DataType_FLOAT8E5M2;
  type_str_to_tensor_data_type_["float8e5m2fnuz"] = TensorProto_DataType_FLOAT8E5M2FNUZ;
  type_str_to_tensor_data_type_["float8e8m0"] = TensorProto_DataType_FLOAT8E8M0;
  type_str_to_tensor_data_type_["uint4"] = TensorProto_DataType_UINT4;
  type_str_to_tensor_data_type_["int4"] = TensorProto_DataType_INT4;
  type_str_to_tensor_data_type_["uint2"] = TensorProto_DataType_UINT2;
  type_str_to_tensor_data_type_["int2"] = TensorProto_DataType_INT2;
  type_str_to_tensor_data_type_["float4e2m1"] = TensorProto_DataType_FLOAT4E2M1;
  type_str_to_tensor_data_type_["float6e2m3"] = TensorProto_DataType_FLOAT6E2M3;
  type_str_to_tensor_data_type_["float6e3m2"] = TensorProto_DataType_FLOAT6E3M2;

  for (auto& [type_str, data_type] : type_str_to_tensor_data_type_) {
    tensor_data_type_to_type_str_[data_type] = type_str;
    allowed_data_types_.insert(type_str);
  }
}

} // namespace

} // namespace ONNX_NAMESPACE::Utils
