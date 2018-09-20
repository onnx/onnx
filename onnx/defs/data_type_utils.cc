#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>

#include "data_type_utils.h"

namespace ONNX_NAMESPACE {
namespace Utils {

// Singleton wrapper around allowed data types.
// This implements construct on first use which is needed to ensure
// static objects are initialized before use. Ops registration does not work
// properly without this.
class TypesWrapper final {
 public:
  static TypesWrapper& GetTypesWrapper();

  std::unordered_set<std::string>& GetAllowedDataTypes();

  std::unordered_map<std::string, int>& TypeStrToTensorDataType();

  std::unordered_map<int, std::string>& TensorDataTypeToTypeStr();

  ~TypesWrapper() = default;
  TypesWrapper(const TypesWrapper&) = delete;
  void operator=(const TypesWrapper&) = delete;

 private:
  TypesWrapper();

  std::unordered_map<std::string, int> type_str_to_tensor_data_type_;
  std::unordered_map<int, std::string> tensor_data_type_to_type_str_;
  std::unordered_set<std::string> allowed_data_types_;
};

// Simple class which contains pointers to external string buffer and a size.
// This can be used to track a "valid" range/slice of the string.
// Caller should ensure StringRange is not used after external storage has
// been freed.
class StringRange final {
 public:
  StringRange();
  StringRange(const char* data, size_t size);
  StringRange(const std::string& str);
  StringRange(const char* data);
  const char* Data() const;
  size_t Size() const;
  bool Empty() const;
  char operator[](size_t idx) const;
  void Reset();
  void Reset(const char* data, size_t size);
  void Reset(const std::string& str);
  bool StartsWith(const StringRange& str) const;
  bool EndsWith(const StringRange& str) const;
  bool LStrip();
  bool LStrip(size_t size);
  bool LStrip(StringRange str);
  bool RStrip();
  bool RStrip(size_t size);
  bool RStrip(StringRange str);
  bool LAndRStrip();
  void ParensWhitespaceStrip();
  size_t Find(const char ch) const;

  // These methods provide a way to return the range of the string
  // which was discarded by LStrip(). i.e. We capture the string
  // range which was discarded.
  StringRange GetCaptured();
  void RestartCapture();

 private:
  // data_ + size tracks the "valid" range of the external string buffer.
  const char* data_;
  size_t size_;

  // start_ and end_ track the captured range.
  // end_ advances when LStrip() is called.
  const char* start_;
  const char* end_;
};

std::unordered_map<std::string, TypeProto>&
DataTypeUtils::GetTypeStrToProtoMap() {
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
  if (GetTypeStrToProtoMap().find(typeStr) == GetTypeStrToProtoMap().end()) {
    GetTypeStrToProtoMap()[typeStr] = type_proto;
  }
  return &(GetTypeStrToProtoMap().find(typeStr)->first);
}

DataType DataTypeUtils::ToType(const std::string& type_str) {
  TypeProto type;
  FromString(type_str, type);
  return ToType(type);
}

const TypeProto& DataTypeUtils::ToTypeProto(const DataType& data_type) {
  std::lock_guard<std::mutex> lock(GetTypeStrLock());
  auto it = GetTypeStrToProtoMap().find(*data_type);
  assert(it != GetTypeStrToProtoMap().end());
  return it->second;
}

std::string DataTypeUtils::ToString(
    const TypeProto& type_proto,
    const std::string& left,
    const std::string& right) {
  switch (type_proto.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      // Note: We do not distinguish tensors with zero rank (a shape consisting
      // of an empty sequence of dimensions) here.
      return left + "tensor(" +
          ToDataTypeString(type_proto.tensor_type().elem_type()) + ")" + right;
    }
#ifdef ONNX_ML
    case TypeProto::ValueCase::kSequenceType: {
      return ToString(
          type_proto.sequence_type().elem_type(), left + "seq(", ")" + right);
    }

    case TypeProto::ValueCase::kMapType: {
      std::string map_str =
          "map(" + ToDataTypeString(type_proto.map_type().key_type()) + ",";
      return ToString(
          type_proto.map_type().value_type(), left + map_str, ")" + right);
    }
    case TypeProto::ValueCase::kOpaqueType: {
      static const std::string empty;
      std::string result;
      std::string param_str;
      const auto& op_type = type_proto.opaque_type();
      const auto param_size = op_type.parameters_size();
      for (int p = 0; p < param_size; ++p) {
        const auto& param_proto = op_type.parameters(p);
        param_str.append(
            ToString(param_proto, (p == 0) ? empty : ",", empty));
      }
      result.append("opaque(");
      result.append(op_type.has_domain() ? op_type.domain() : empty)
          .append(",");
      result.append(op_type.has_name() ? op_type.name() : empty)
          .append(",");
      result.append("p(").append(param_str).append("))");
      return result;
    }
#endif
    default:
      assert(false);
      return std::string();
  }
}

std::string DataTypeUtils::ToDataTypeString(
    const TensorProto::DataType& tensor_data_type) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  auto iter = t.TensorDataTypeToTypeStr().find(tensor_data_type);
  assert(t.TensorDataTypeToTypeStr().end() != iter);
  return iter->second;
}

void DataTypeUtils::FromString(
    const std::string& type_str,
    TypeProto& type_proto) {
  StringRange s(type_str);
  type_proto.Clear();
#ifdef ONNX_ML
  if (s.LStrip("seq")) {
    s.ParensWhitespaceStrip();
    return FromString(
        std::string(s.Data(), s.Size()),
        *type_proto.mutable_sequence_type()->mutable_elem_type());
  } else if (s.LStrip("map")) {
    s.ParensWhitespaceStrip();
    size_t key_size = s.Find(',');
    StringRange k(s.Data(), key_size);
    std::string key(k.Data(), k.Size());
    s.LStrip(key_size);
    s.LStrip(",");
    StringRange v(s.Data(), s.Size());
    TensorProto::DataType key_type;
    FromDataTypeString(key, key_type);
    type_proto.mutable_map_type()->set_key_type(key_type);
    return FromString(
        std::string(v.Data(), v.Size()),
        *type_proto.mutable_map_type()->mutable_value_type());
  } else if (s.LStrip("opaque(")) {
    // Possible to have an Opaque type w/o parameters
    // so we make sure it exists
    auto* opaque_type = type_proto.mutable_opaque_type();
    s.ParensWhitespaceStrip();
    // Check if we have domain which is positionally first
    size_t cm = s.Find(',');
    if (cm > 0) {
      opaque_type->mutable_domain()->assign(s.Data(), cm);
    }
    s.LStrip(cm + 1);
    cm = s.Find(',');
    if (cm > 0) {
      opaque_type->mutable_name()->assign(s.Data(), cm);
    }
    s.LStrip(cm + 1);
    if (s.LStrip("p(")) {
      s.ParensWhitespaceStrip();
      auto param_size = s.Find(',');
      while (param_size != std::string::npos) {
        // Means there are multiple params
        std::string param(s.Data(), param_size);
        FromString(param, *opaque_type->add_parameters());
        s.LStrip(param_size);
        s.LStrip(",");
        param_size = s.Find(',');
      }
      if (!s.Empty()) {
        std::string param(s.Data(), s.Size());
        FromString(param, *opaque_type->add_parameters());
      }
    }
  } else
#endif
  if (s.LStrip("tensor")) {
    s.ParensWhitespaceStrip();
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    type_proto.mutable_tensor_type()->set_elem_type(e);
  } else {
    // Scalar
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    TypeProto::Tensor* t = type_proto.mutable_tensor_type();
    t->set_elem_type(e);
    // Call mutable_shape() to initialize a shape with no dimension.
    t->mutable_shape();
  }
}

bool DataTypeUtils::IsValidDataTypeString(const std::string& type_str) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  const auto& allowedSet = t.GetAllowedDataTypes();
  return (allowedSet.find(type_str) != allowedSet.end());
}

void DataTypeUtils::FromDataTypeString(
    const std::string& type_str,
    TensorProto::DataType& tensor_data_type) {
  assert(IsValidDataTypeString(type_str));

  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  tensor_data_type =
      (TensorProto::DataType)t.TypeStrToTensorDataType()[type_str];
}

StringRange::StringRange() : data_(""), size_(0), start_(data_), end_(data_) {}

StringRange::StringRange(const char* p_data, size_t p_size)
    : data_(p_data), size_(p_size), start_(data_), end_(data_) {
  assert(p_data != nullptr);
  LAndRStrip();
}

StringRange::StringRange(const std::string& p_str)
    : data_(p_str.data()), size_(p_str.size()), start_(data_), end_(data_) {
  LAndRStrip();
}

StringRange::StringRange(const char* p_data)
    : data_(p_data), size_(strlen(p_data)), start_(data_), end_(data_) {
  LAndRStrip();
}

const char* StringRange::Data() const {
  return data_;
}

size_t StringRange::Size() const {
  return size_;
}

bool StringRange::Empty() const {
  return size_ == 0;
}

char StringRange::operator[](size_t idx) const {
  return data_[idx];
}

void StringRange::Reset() {
  data_ = "";
  size_ = 0;
  start_ = end_ = data_;
}

void StringRange::Reset(const char* data, size_t size) {
  data_ = data;
  size_ = size;
  start_ = end_ = data_;
}

void StringRange::Reset(const std::string& str) {
  data_ = str.data();
  size_ = str.size();
  start_ = end_ = data_;
}

bool StringRange::StartsWith(const StringRange& str) const {
  return ((size_ >= str.size_) && (memcmp(data_, str.data_, str.size_) == 0));
}

bool StringRange::EndsWith(const StringRange& str) const {
  return (
      (size_ >= str.size_) &&
      (memcmp(data_ + (size_ - str.size_), str.data_, str.size_) == 0));
}

bool StringRange::LStrip() {
  size_t count = 0;
  const char* ptr = data_;
  while (count < size_ && isspace(*ptr)) {
    count++;
    ptr++;
  }

  if (count > 0) {
    return LStrip(count);
  }
  return false;
}

bool StringRange::LStrip(size_t size) {
  if (size <= size_) {
    data_ += size;
    size_ -= size;
    end_ += size;
    return true;
  }
  return false;
}

bool StringRange::LStrip(StringRange str) {
  if (StartsWith(str)) {
    return LStrip(str.size_);
  }
  return false;
}

bool StringRange::RStrip() {
  size_t count = 0;
  const char* ptr = data_ + size_ - 1;
  while (count < size_ && isspace(*ptr)) {
    ++count;
    --ptr;
  }

  if (count > 0) {
    return RStrip(count);
  }
  return false;
}

bool StringRange::RStrip(size_t size) {
  if (size_ >= size) {
    size_ -= size;
    return true;
  }
  return false;
}

bool StringRange::RStrip(StringRange str) {
  if (EndsWith(str)) {
    return RStrip(str.size_);
  }
  return false;
}

bool StringRange::LAndRStrip() {
  bool l = LStrip();
  bool r = RStrip();
  return l || r;
}

void StringRange::ParensWhitespaceStrip() {
  LStrip();
  LStrip("(");
  LAndRStrip();
  RStrip(")");
  RStrip();
}

size_t StringRange::Find(const char ch) const {
  size_t idx = 0;
  while (idx < size_) {
    if (data_[idx] == ch) {
      return idx;
    }
    idx++;
  }
  return std::string::npos;
}

void StringRange::RestartCapture() {
  start_ = data_;
  end_ = data_;
}

StringRange StringRange::GetCaptured() {
  return StringRange(start_, end_ - start_);
}

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

  for (auto& str_type_pair : type_str_to_tensor_data_type_) {
    tensor_data_type_to_type_str_[str_type_pair.second] = str_type_pair.first;
    allowed_data_types_.insert(str_type_pair.first);
  }
}
} // namespace Utils
} // namespace ONNX_NAMESPACE
