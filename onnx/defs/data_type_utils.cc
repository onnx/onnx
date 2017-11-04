#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>

#include "data_type_utils.h"

namespace onnx {
namespace Utils {
// Simple class which contains pointers to external string buffer and a size.
// This can be used to track a "valid" range/slice of the string.
// Caller should ensure StringRange is not used after external storage has
// been freed.
class StringRange {
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
      if (type_proto.tensor_type().has_shape() &&
          type_proto.tensor_type().shape().dim_size() == 0) {
        // Scalar case.
        return left + ToDataTypeString(type_proto.tensor_type().elem_type()) +
            right;
      } else {
        return left + "tensor(" +
            ToDataTypeString(type_proto.tensor_type().elem_type()) + ")" + right;
      }
    }

    case TypeProto::ValueCase::kSparseTensorType:
      return left + "sparse(" +
          ToDataTypeString(type_proto.sparse_tensor_type().elem_type()) + ")" +
          right;

    default:
      assert(false);
      return "";
  }
}

std::string DataTypeUtils::ToDataTypeString(
    const TensorProto::DataType& tensor_data_type) {
  TypesWrapper& t = TypesWrapper::GetTypesWrapper();
  switch (tensor_data_type) {
    case TensorProto::DataType::TensorProto_DataType_BOOL:
      return t.kBool;
    case TensorProto::DataType::TensorProto_DataType_STRING:
      return t.kString;
    case TensorProto::DataType::TensorProto_DataType_FLOAT16:
      return t.kFloat16;
    case TensorProto::DataType::TensorProto_DataType_FLOAT:
      return t.kFloat;
    case TensorProto::DataType::TensorProto_DataType_DOUBLE:
      return t.kDouble;
    case TensorProto::DataType::TensorProto_DataType_INT8:
      return t.kInt8;
    case TensorProto::DataType::TensorProto_DataType_INT16:
      return t.kInt16;
    case TensorProto::DataType::TensorProto_DataType_INT32:
      return t.kInt32;
    case TensorProto::DataType::TensorProto_DataType_INT64:
      return t.kInt64;
    case TensorProto::DataType::TensorProto_DataType_UINT8:
      return t.kUint8;
    case TensorProto::DataType::TensorProto_DataType_UINT16:
      return t.kUint16;
    case TensorProto::DataType::TensorProto_DataType_UINT32:
      return t.kUint32;
    case TensorProto::DataType::TensorProto_DataType_UINT64:
      return t.kUint64;
    case TensorProto::DataType::TensorProto_DataType_COMPLEX64:
      return t.kComplex64;
    case TensorProto::DataType::TensorProto_DataType_COMPLEX128:
      return t.kComplex128;
  }

  assert(false);
  return "";
}

void DataTypeUtils::FromString(const std::string& type_str, TypeProto& type_proto) {
  StringRange s(type_str);
  type_proto.Clear();
  if (s.LStrip("sparse")) {
    s.ParensWhitespaceStrip();
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    type_proto.mutable_sparse_tensor_type()->set_elem_type(e);
  } else if (s.LStrip("tensor")) {
    s.ParensWhitespaceStrip();
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    type_proto.mutable_tensor_type()->set_elem_type(e);
  } else {
    // Scalar
    TensorProto::DataType e;
    FromDataTypeString(std::string(s.Data(), s.Size()), e);
    TypeProto::TensorTypeProto* t = type_proto.mutable_tensor_type();
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
  if (type_str == t.kBool) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_BOOL;
  } else if (type_str == t.kFloat) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_FLOAT;
  } else if (type_str == t.kFloat16) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_FLOAT16;
  } else if (type_str == t.kDouble) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_DOUBLE;
  } else if (type_str == t.kInt8) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_INT8;
  } else if (type_str == t.kInt16) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_INT16;
  } else if (type_str == t.kInt32) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_INT32;
  } else if (type_str == t.kInt64) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_INT64;
  } else if (type_str == t.kString) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_STRING;
  } else if (type_str == t.kUint8) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_UINT8;
  } else if (type_str == t.kUint16) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_UINT16;
  } else if (type_str == t.kUint32) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_UINT32;
  } else if (type_str == t.kUint64) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_UINT64;
  } else if (type_str == t.kComplex64) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_COMPLEX64;
  } else if (type_str == t.kComplex128) {
    tensor_data_type = TensorProto::DataType::TensorProto_DataType_COMPLEX128;
  } else {
    assert(false);
  }
}

StringRange::StringRange()
    : data_(""), size_(0), start_(data_), end_(data_) {}

StringRange::StringRange(const char* p_data, size_t p_size)
    : data_(p_data), size_(p_size), start_(data_), end_(data_) {
  assert(p_data != nullptr);
  LAndRStrip();
}

StringRange::StringRange(const std::string& p_str)
    : data_(p_str.data()),
      size_(p_str.size()),
      start_(data_),
      end_(data_) {
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
  return (
      (size_ >= str.size_) &&
      (memcmp(data_, str.data_, str.size_) == 0));
}

bool StringRange::EndsWith(const StringRange& str) const {
  return (
      (size_ >= str.size_) &&
      (memcmp(data_ + (size_ - str.size_), str.data_, str.size_) ==
       0));
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
  static std::unordered_set<std::string> allowedDataTypes = {kFloat16,
                                                             kFloat,
                                                             kDouble,
                                                             kInt8,
                                                             kInt16,
                                                             kInt32,
                                                             kInt64,
                                                             kUint8,
                                                             kUint16,
                                                             kUint32,
                                                             kUint64,
                                                             kComplex64,
                                                             kComplex128,
                                                             kString,
                                                             kBool};
  return allowedDataTypes;
}
} // namespace Utils
} // namespace onnx
