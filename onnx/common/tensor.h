// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/onnx_pb.h"
#include <math.h>
#include <numeric>

namespace ONNX_NAMESPACE {

struct Tensor final {
private:
  bool is_segment_;
  int64_t segment_begin_;
  int64_t segment_end_;
  bool has_name_;
  std::string name_;
  ONNX_NAMESPACE::TensorProto_DataType elem_type_;
  std::vector<int64_t> sizes_;

  std::vector<float> float_data_;
  std::vector<double> double_data_;
  std::vector<int32_t> int32_data_;
  std::vector<int64_t> int64_data_;
  std::vector<uint64_t> uint64_data_;
  std::vector<std::string> string_data_;

  bool is_raw_data_;
  std::string raw_data_;

  template<typename T>
  static void add_nums(T* x, const T* y) {
    *x += *y;
  }

  template<typename T>
  static void sub_nums(T* x, const T* y) {
    *x -= *y;
  }

  template<typename T>
  static void mult_nums(T* x, const T* y) {
    *x *= *y;
  }

  template<typename T>
  static void divide_nums(T* x, const T* y) {
    *x /= *y;
  }

  template<typename T>
  static void sqrt_num(T* x) {
    *x = (T) std::sqrt((double) *x);
  }


  template<typename T, int64_t block_size>
  void bin_func(void (*f)(T*, const T*), Tensor& a, std::vector<T>& T_data_, std::vector<T>& a_T_data_) {
    int64_t num_elements = std::accumulate(sizes_.begin(), sizes_.end(), (int64_t) 1, std::multiplies<int64_t>());
    T* T_ptr;
    const T* a_ptr;
    if (is_raw_data_)  {
      T_data_.clear();
      for (auto i = 0; i < raw_data_.size(); i += sizeof(T))  {
        T_data_.push_back(*((T*)(raw_data_.c_str() + i)));
      }
    }
    T_ptr = (T*) T_data_.data();
    if (a.is_raw_data())  {
      a_ptr = (const T*) a.raw().c_str();
    } else {
      a_ptr = (const T*) a_T_data_.data();
    }
    for (int64_t i = 0; i < num_elements; i++) {
      f(T_ptr + i * block_size, a_ptr + i * block_size);
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
    }
  }

  template<typename T, int64_t block_size>
  void un_func(void (*f)(T*), std::vector<T>& T_data_)  {
    int64_t num_elements = std::accumulate(sizes_.begin(), sizes_.end(), (int64_t) 1, std::multiplies<int64_t>());
    T* T_ptr;
    if (is_raw_data_)  {
      T_data_.clear();
      for (auto i = 0; i < raw_data_.size(); i += sizeof(T))  {
        T_data_.push_back(*((T*)(raw_data_.c_str() + i)));
      }
    }
    T_ptr = (T*) T_data_.data();
    for (int64_t i = 0; i < num_elements; i++) {
      f(T_ptr + i * block_size);
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
    }
  }

  template<typename T, int64_t block_size>
  void scale_dim(Tensor&s, std::vector<T>& T_data_)  {
    int64_t elems_per_first_dim = std::accumulate(sizes_.begin() + 1, sizes_.end(), block_size, std::multiplies<int64_t>());
    int64_t first_dim_size = sizes_[0];
    T* T_ptr;
    const T* scales = (const T*) s.raw().c_str();
    if (is_raw_data_)  {
      T_data_.clear();
      for (auto i = 0; i < raw_data_.size(); i += sizeof(T))  {
        T_data_.push_back(*((T*)(raw_data_.c_str() + i)));
      }
    }
    T_ptr = (T*) T_data_.data();
    int64_t counter = 0;
    for (int64_t i = 0; i < first_dim_size; i++)  {
      for (int64_t j = 0; j < elems_per_first_dim; j++) {
        T_ptr[counter++] *= scales[i];
      }
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
    }
  }

  void check(Tensor &a) {
    if (a.elem_type() != elem_type_) {
      throw("Type of tensors do not match");
    }
    if (a.sizes() != sizes_) {
      throw("Tensor shapes are incompatible");
    }
  }

public:
  Tensor()
  : is_segment_(false)
  , segment_begin_(0)
  , segment_end_(0)
  , has_name_(false)
  , elem_type_(ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED)
  , is_raw_data_(false)
  {}

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }
  std::vector<int64_t>& sizes() {
    return sizes_;
  }

  ONNX_NAMESPACE::TensorProto_DataType elem_type() const {
    return elem_type_;
  }

  ONNX_NAMESPACE::TensorProto_DataType& elem_type() {
    return elem_type_;
  }

  std::vector<std::string>& strings() {
    return string_data_;
  }

  const std::vector<std::string>& strings() const {
    return string_data_;
  }

  std::vector<float>& floats() {
    return float_data_;
  }

  const std::vector<float>& floats() const {
    return float_data_;
  }

  std::vector<double>& doubles() {
    return double_data_;
  }

  const std::vector<double>& doubles() const {
    return double_data_;
  }

  std::vector<int32_t>& int32s() {
    return int32_data_;
  }

  const std::vector<int32_t>& int32s() const {
    return int32_data_;
  }

  std::vector<int64_t>& int64s() {
    return int64_data_;
  }

  const std::vector<int64_t>& int64s() const {
    return int64_data_;
  }

  std::vector<uint64_t>& uint64s() {
    return uint64_data_;
  }

  const std::vector<uint64_t>& uint64s() const {
    return uint64_data_;
  }

  const std::string& raw() const {
    return raw_data_;
  }

  void set_raw_data(std::string raw_data) {
    is_raw_data_ = true;
    raw_data_ = std::move(raw_data);
  }

  bool is_segment() const {
    return is_segment_;
  }

  int64_t segment_begin() const {
    return segment_begin_;
  }

  int64_t segment_end() const {
    return segment_end_;
  }

  void set_segment_begin_and_end(int64_t begin, int64_t end) {
    is_segment_ = true;
    segment_begin_ = begin;
    segment_end_ = end;
  }

  bool hasName() const {
    return has_name_;
  }

  const std::string& name() const {
    return name_;
  }

  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }

  bool is_raw_data() const {
    return is_raw_data_;
  }

  //applies function f element-wise to this, storing result in this
  //type T must correspond to appropriate tensor type

  void apply_unary_function(void (*f)(float*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        un_func<float, 1>(f, float_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
        un_func<float, 2>(f, float_data_);
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_unary_function(void (*f)(int32_t*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
        un_func<int32_t, 1>(f, int32_data_);
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_unary_function(void (*f)(int64_t*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        un_func<int64_t, 1>(f, int64_data_);
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_unary_function(void (*f)(uint64_t*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
        un_func<uint64_t, 1>(f, uint64_data_);
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_unary_function(void (*f)(double*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {
        un_func<double, 1>(f, double_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:  {
        un_func<double, 2>(f, double_data_);
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  //applies function f element-wise to this and a, storing result in this
  //type T must correspond to appropriate tensor type

  void apply_binary_function(void (*f)(float*, const float*), Tensor& a)  {
    check(a);
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        bin_func<float, 1>(f, a, float_data_, a.floats());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
        bin_func<float, 2>(f, a, float_data_, a.floats());
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_binary_function(void (*f)(int32_t*, const int32_t*), Tensor& a)  {
    check(a);
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
        bin_func<int32_t, 1>(f, a, int32_data_, a.int32s());
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_binary_function(void (*f)(int64_t*, const int64_t*), Tensor& a)  {
    check(a);
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        bin_func<int64_t, 1>(f, a, int64_data_, a.int64s());
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_binary_function(void (*f)(uint64_t*, const uint64_t*), Tensor& a)  {
    check(a);
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
        bin_func<uint64_t, 1>(f, a, uint64_data_, a.uint64s());
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  void apply_binary_function(void (*f)(double*, const double*), Tensor& a)  {
    check(a);
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:  {
        bin_func<double, 1>(f, a, double_data_, a.doubles());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:  {
        bin_func<double, 2>(f, a, double_data_, a.doubles());
        break;
      }
      default:
        throw("Operation not supported for this data type");
    }
  }

  //this += a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE,
  //TODO: Support for COMPLEX64, COMPLEX128
  void add(Tensor& a);

  //this -= a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for COMPLEX64, COMPLEX128
  void subtract(Tensor& a);

  //this *= a
  //Supported for
  //FLOAT, BOOL, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void multiply(Tensor& a);

  //this /= a
  //Supported for
  //FLOAT, INT8, INT16, INT32, UINT8, UINT16, INT64,
  //UINT32, UINT64, DOUBLE
  //TODO: Support for FLOAT16, COMPLEX64, COMPLEX128
  void divide(Tensor& a);

  //Element-wise square root of This
  //Supported for
  //FLOAT, DOUBLE,
  //TODO: Support for FLOAT16
  void sqrt();

  //Element wise scaling of tensor s
  //s is one dimensional, has size M, where M is size of first dimension of tensor
  //s must have data as raw_data and has data type corresponding to this
  void scale_by_first_dim(Tensor& s);
};

inline void Tensor::add(Tensor& a) {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
      apply_binary_function(&(add_nums<float>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
      apply_binary_function(&(add_nums<int32_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
      apply_binary_function(&(add_nums<int64_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
      apply_binary_function(&(add_nums<uint64_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:  {
      apply_binary_function(&(add_nums<double>), a);
      break;
    }
    default:
      throw("Operation not supported for this data type");
  }

}

inline void Tensor::subtract(Tensor& a) {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
      apply_binary_function(&(sub_nums<float>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
      apply_binary_function(&(sub_nums<int32_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
      apply_binary_function(&(sub_nums<int64_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
      apply_binary_function(&(sub_nums<uint64_t>), a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:  {
      apply_binary_function(&(sub_nums<double>), a);
      break;
    }
    default:
      throw("Operation not supported for this data type");
  }

}

inline void Tensor::multiply(Tensor& a) {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      apply_binary_function(mult_nums<float>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
      apply_binary_function(mult_nums<int32_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
      apply_binary_function(mult_nums<int64_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
      apply_binary_function(mult_nums<uint64_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      apply_binary_function(mult_nums<double>, a);
      break;
    }
    default:
      throw("Operation not supported for this data type");
  }
}

inline void Tensor::divide(Tensor& a) {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      apply_binary_function(divide_nums<float>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
      apply_binary_function(divide_nums<int32_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
      apply_binary_function(divide_nums<int64_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
      apply_binary_function(divide_nums<uint64_t>, a);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      apply_binary_function(divide_nums<double>, a);
      break;
    }
    default:
      throw("Operation not supported for this data type");
  }
}

inline void Tensor::sqrt() {
  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      apply_unary_function(sqrt_num<float>);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      apply_unary_function(sqrt_num<double>);
      break;
    }
    default:
      throw("Operation not supported for this data type");
  }
}

inline void Tensor::scale_by_first_dim(Tensor& s) {
  ONNX_ASSERT(sizes_.size() > 1 && s.sizes().size() == 1 && s.sizes()[0] == sizes_[0]);
  ONNX_ASSERT(s.is_raw_data() && s.elem_type() == elem_type_);

  switch(elem_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
      scale_dim<float, 1>(s, float_data_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
      scale_dim<float, 2>(s, float_data_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      scale_dim<int32_t, 1>(s, int32_data_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      scale_dim<double, 1>(s, double_data_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
      scale_dim<double, 2>(s, double_data_);
      break;
    }
    default:
      throw("Incompatible data type: FLOAT, COMPLEX64, FLOAT16, DOUBLE and COMPLEX128 supported");
  }
}

} // namespace ONNX_NAMESPACE
