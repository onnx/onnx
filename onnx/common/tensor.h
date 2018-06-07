// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/onnx_pb.h"

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

  template<typename T, int64_t block_size>
  void bin_func(void (*f)(void*, const void*), Tensor& a, std::vector<T>& T_data_, std::vector<T>& a_T_data_) {
    T* T_ptr;
    const T* a_ptr;
    if (is_raw_data_)  {
      T_ptr = (T*) malloc(raw_data_.size());
      memcpy(T_ptr, raw_data_.c_str(), raw_data_.size());
    } else {
      T_ptr = (T*) &T_data_[0];
    }
    if (a.is_raw_data())  {
      a_ptr = (const T*) a.raw().c_str();
    } else {
      a_ptr = (const T*) &a_T_data_[0];
    }
    for (int i = 0; i < T_data_.size(); i++) {
      f((void*)(T_ptr + i * block_size), (const void*)(a_ptr + i * block_size));
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
      free(T_ptr);
    }
  }

  template<typename T, int64_t block_size>
  void un_func(void (*f)(void*), std::vector<T>& T_data_)  {
    T* T_ptr;
    if (is_raw_data_)  {
      T_ptr = (T*) malloc(raw_data_.size());
      memcpy(T_ptr, raw_data_.c_str(), raw_data_.size());
    } else {
      T_ptr = (T*) &T_data_[0];
    }
    for (int i = 0; i < T_data_.size(); i++) {
      f((void*)(T_ptr + i * block_size));
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
      free(T_ptr);
    }
  }

  template<typename T, int64_t block_size>
  void channel_scale(Tensor&s, std::vector<T>& T_data_)  {
    int64_t dim_per_data = block_size;
    for (int i = 2; i < sizes_.size(); i++) {
      dim_per_data *= sizes_[i];
    }
    int64_t feature_maps = sizes_[0];
    int64_t num_channels = sizes_[1];
    T* T_ptr;
    const T* channel_scales = (const T*) s.raw().c_str();
    if (is_raw_data_)  {
      T_ptr = (T*) malloc(raw_data_.size());
      memcpy(T_ptr, raw_data_.c_str(), raw_data_.size());
    } else {
      T_ptr = (T*) &T_data_[0];
    }
    int counter = 0;
    for (int i = 0; i < feature_maps; i++)  {
      for(int j = 0; j < num_channels; j++) {
        for (int k = 0; k < dim_per_data; k++)  {
          T_ptr[counter++] *= channel_scales[j];
        }
      }
    }
    if (is_raw_data_)  {
      raw_data_.assign((const char*) T_ptr, raw_data_.size());
      free(T_ptr);
    }
  }



  //Element wise scaling of tensor by scale_by_channel
  //s is one dimensional, has size C, where C is number of channels
  //s must have data as raw_data and has data type corresponding to this
  void scale_by_channel(Tensor& s) {
    ONNX_ASSERT(sizes_.size() > 2 && s.sizes().size() == 1 && s.sizes()[0] == sizes_[1]);
    ONNX_ASSERT(s.is_raw_data() && s.elem_type() == elem_type_);

    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        channel_scale<float, 1>(s, float_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        channel_scale<float, 2>(s, float_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        channel_scale<int32_t, 1>(s, int32_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        channel_scale<double, 1>(s, double_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
        channel_scale<double, 2>(s, double_data_);
        break;
      }
      default:
        throw("Incompatible data type: FLOAT, COMPLEX64, FLOAT16, DOUBLE and COMPLEX128 supported");
    }
  }



  //applies function f element-wise to this and a, storing result in this
  //WARNING: does not type check, so ensure that the tensors have the correct
  //types before using
  void apply_binary_function(void (*f)(void*, const void*), Tensor& a)  {
    if (a.elem_type() != elem_type_) {
      throw("Type of tensors do not match");
    }
    if (a.sizes() != sizes_) {
      throw("Tensor shapes are incompatible");
    }

    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        bin_func<float, 1>(f, a, float_data_, a.floats());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
        bin_func<float, 2>(f, a, float_data_, a.floats());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
        bin_func<int32_t, 1>(f, a, int32_data_, a.int32s());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        bin_func<int64_t, 1>(f, a, int64_data_, a.int64s());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
        bin_func<uint64_t, 1>(f, a, uint64_data_, a.uint64s());
        break;
      }
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

  //applies function f element-wise to this, storing result in this
  //WARNING: does not type check, so ensure that this tensor has the correct
  //type before using
  void apply_unary_function(void (*f)(void*))  {
    switch(elem_type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:  {
        un_func<float, 1>(f, float_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:  {
        un_func<float, 2>(f, float_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:  {
        un_func<int32_t, 1>(f, int32_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:  {
        un_func<int64_t, 1>(f, int64_data_);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:  {
        un_func<uint64_t, 1>(f, uint64_data_);
        break;
      }
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
};

} // namespace ONNX_NAMESPACE
