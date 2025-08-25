// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/onnx2/cpu/stream.h"

#include <stdint.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx2 {
namespace utils {

///////////////
// BinaryStream
///////////////

std::string FieldNumber::string() const {
  return common_helpers::MakeString("[field_number=", field_number, ", wire_type=", wire_type, "]");
}

void BinaryStream::_check() {
  EXT_ENFORCE(limits_.empty(), "BinaryStream destructor called with non-empty limits stack.");
}

BinaryStream::~BinaryStream() {
  _check();
}

RefString BinaryStream::next_string() {
  // Depending on the stream implementation, the string may be disappear after reading another item.
  uint64_t length = next_uint64();
  this->CanRead(length, "[StringStream::next_string]");
  return RefString(reinterpret_cast<const char*>(read_bytes(length)), static_cast<size_t>(length));
}

int64_t BinaryStream::next_int64() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int64_t>(value);
}

int32_t BinaryStream::next_int32() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int32_t>(value);
}

float BinaryStream::next_float() {
  float value;
  read_bytes(sizeof(float), reinterpret_cast<uint8_t*>(&value));
  return value;
}

double BinaryStream::next_double() {
  double value;
  read_bytes(sizeof(double), reinterpret_cast<uint8_t*>(&value));
  return value;
}

FieldNumber BinaryStream::next_field() {
  FieldNumber n;
  n.wire_type = next_uint64();
  n.field_number = n.wire_type >> 3;
  n.wire_type = n.wire_type & 0x07;
  return n;
}

void BinaryStream::ReadDelayedBlock(DelayedBlock&) {
  EXT_THROW("ReadDelayedBlock is not implemented for this stream.");
}

void BinaryStream::WaitForDelayedBlock() {
  EXT_THROW("WaitForDelayedBlock is not implemented for this stream.");
}

void BinaryStream::StartThreadPool(size_t) {
  EXT_THROW("StartThreadPool is not implemented for this stream.");
}

void BinaryStream::LimitToNext(uint64_t length) {
  CanRead(length, "Too many bytes requested in LimitToNext.");
  limits_.push_back(size());
  LimitTo(tell() + length);
}

void BinaryStream::Restore() {
  EXT_ENFORCE(!limits_.empty(), "Cannot restore, no limits set");
  uint64_t last_limit = limits_.back();
  LimitTo(last_limit);
  limits_.pop_back();
}

///////////////
// StringStream
///////////////

void StringStream::Setup(const uint8_t* data, int64_t size) {
  EXT_ENFORCE(!thread_pool_.IsStarted(), "ThreadPool is still running.");
  pos_ = 0;
  size_ = size;
  data_ = data;
  thread_pool_.Clear();
  blocks_.clear();
}

void StringStream::CanRead(uint64_t len, const char* msg) {
  EXT_ENFORCE(
      pos_ + static_cast<int64_t>(len) <= size_,
      msg,
      " unable to read ",
      len,
      " bytes, pos_=",
      pos_,
      ", size_=",
      size_);
}

const uint8_t* StringStream::read_bytes(offset_t n_bytes, uint8_t* pre_allocated_buffer) {
  if (pre_allocated_buffer != nullptr) {
    memcpy(pre_allocated_buffer, data_ + pos_, n_bytes);
    pos_ += n_bytes;
    return pre_allocated_buffer;
  } else {
    const uint8_t* res = data_ + pos_;
    pos_ += n_bytes;
    return res;
  }
}

void StringStream::skip_bytes(offset_t n_bytes) {
  pos_ += n_bytes;
}

uint64_t StringStream::next_uint64() {
  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10 && pos_ < size_; ++i) {
    uint8_t byte = data_[pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[StringStream::next_uint64] unable to read an uint64 at pos=", pos_, ", size=", size_);
}

std::string StringStream::tell_around() const {
  offset_t begin = pos_;
  offset_t end = pos_ + 10 < static_cast<offset_t>(size()) ? pos_ + 10 : static_cast<offset_t>(size());
  RefString ref(reinterpret_cast<const char*>(data_) + begin, end - begin);
  return ref.as_string();
}

void StringStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void StringStream::ReadDelayedBlock(DelayedBlock& block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0, "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask([this, block]() { memcpy(block.data, this->data_ + block.offset, block.size); });
  pos_ += block.size;
}

void StringStream::WaitForDelayedBlock() {
  thread_pool_.Wait();
}

void StringStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(n_threads);
}

////////////////////
// BinaryWriteStream
////////////////////

void BinaryWriteStream::write_variant_uint64(uint64_t value) {
  uint8_t v;
  while (value > 127) {
    v = static_cast<uint8_t>((value & 0x7F) | 0x80);
    write_raw_bytes(&v, 1);
    value >>= 7;
  }
  v = static_cast<uint8_t>(value);
  write_raw_bytes(reinterpret_cast<uint8_t*>(&v), 1);
}

uint64_t BinaryWriteStream::size_variant_uint64(uint64_t value) {
  return VarintSize(value);
}

void BinaryWriteStream::write_int64(int64_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int64(int64_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_int32(int32_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int32(int32_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_float(float value) {
  write_raw_bytes(reinterpret_cast<uint8_t*>(&value), sizeof(float));
}

uint64_t BinaryWriteStream::size_float(float) {
  return sizeof(float);
}

void BinaryWriteStream::write_double(double value) {
  write_raw_bytes(reinterpret_cast<uint8_t*>(&value), sizeof(double));
}

uint64_t BinaryWriteStream::size_double(double) {
  return sizeof(double);
}

void BinaryWriteStream::write_field_header(uint32_t field_number, uint8_t wire_type) {
  write_variant_uint64((field_number << 3) | wire_type);
}

uint64_t BinaryWriteStream::VarintSize(uint64_t value) {
  size_t size = 0;
  do {
    size++;
    value >>= 7;
  } while (value != 0);
  return size;
}

uint64_t BinaryWriteStream::size_field_header(uint32_t field_number, uint8_t wire_type) {
  return VarintSize((field_number << 3) | wire_type);
}

void BinaryWriteStream::write_string(const std::string& value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const std::string& value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const String& value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const String& value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const RefString& value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const RefString& value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string_stream(const StringWriteStream& stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const StringWriteStream& stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::write_string_stream(const BorrowedWriteStream& stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const BorrowedWriteStream& stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::CacheSize(const void* ptr, uint64_t size) {
  size_cache_[ptr] = size;
}

bool BinaryWriteStream::GetCachedSize(const void* ptr, uint64_t& size) {
  auto it = size_cache_.find(ptr);
  if (it != size_cache_.end()) {
    size = it->second;
    return true;
  }
  return false;
}

////////////////////
// StringWriteStream
////////////////////

void StringWriteStream::write_raw_bytes(const uint8_t* ptr, offset_t n_bytes) {
  buffer_.insert(buffer_.end(), ptr, ptr + n_bytes);
}

int64_t StringWriteStream::size() const {
  return buffer_.size();
}
const uint8_t* StringWriteStream::data() const {
  return buffer_.data();
}

//////////////////////
// BorrowedWriteStream
//////////////////////

void BorrowedWriteStream::write_raw_bytes(const uint8_t*, offset_t) {
  EXT_THROW("This method cannot be called on this class (BorrowedWriteStream).");
}

//////////////////
// FileWriteStream
//////////////////

FileWriteStream::FileWriteStream(const std::string& file_path)
    : BinaryWriteStream(), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  written_bytes_ = 0;
}

void FileWriteStream::write_raw_bytes(const uint8_t* data, offset_t n_bytes) {
  file_stream_.write(reinterpret_cast<const char*>(data), n_bytes);
  written_bytes_ += static_cast<uint64_t>(n_bytes);
}

int64_t FileWriteStream::size() const {
  return static_cast<int64_t>(written_bytes_);
}

const uint8_t* FileWriteStream::data() const {
  EXT_THROW("This method cannot be called on this class (FileWriteStream).");
}

/////////////
// FileStream
/////////////

FileStream::FileStream(const std::string& file_path)
    : lock_(false), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  if (!file_stream_.is_open()) {
    EXT_THROW("Unable to open file: ", file_path);
  }
  file_stream_.seekg(0, std::ios::end);
  std::streampos end = file_stream_.tellg();
  file_stream_.seekg(0);
  size_ = static_cast<offset_t>(end);
}

bool FileStream::is_open() const {
  return file_stream_.is_open();
}

void FileStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void FileStream::CanRead(uint64_t len, const char* msg) {
  EXT_ENFORCE(
      static_cast<int64_t>(tell()) + static_cast<int64_t>(len) <= size_,
      msg,
      " unable to read ",
      len,
      " bytes, pos_=",
      tell(),
      ", size_=",
      size_);
}

uint64_t FileStream::next_uint64() {
  uint64_t result = 0;
  int shift = 0;

  uint8_t byte;
  for (int i = 0; i < 10; ++i) {
    read_bytes(1, &byte);
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[FileStream::next_uint64] unable to read an int64 at pos=", tell(), ", size=", size_);
}

const uint8_t* FileStream::read_bytes(offset_t n_bytes, uint8_t* pre_allocated_buffer) {
  if (pre_allocated_buffer) {
    file_stream_.read(reinterpret_cast<char*>(pre_allocated_buffer), n_bytes);
    return pre_allocated_buffer;
  }
  if (n_bytes > static_cast<offset_t>(buffer_.size()))
    buffer_.resize(n_bytes);
  file_stream_.read(reinterpret_cast<char*>(buffer_.data()), n_bytes);
  return buffer_.data();
}

void FileStream::skip_bytes(offset_t n_bytes) {
  file_stream_.seekg(n_bytes, std::ios::cur);
}

bool FileStream::NotEnd() const {
  return static_cast<int64_t>(tell()) < size_;
}

offset_t FileStream::tell() const {
  return static_cast<offset_t>(const_cast<std::ifstream&>(file_stream_).tellg());
}

std::string FileStream::tell_around() const {
  RefString ref(reinterpret_cast<const char*>(buffer_.data()), buffer_.size() < 10 ? buffer_.size() : 10);
  return ref.as_string();
}

FileStream::~FileStream() {}

void FileStream::ReadDelayedBlock(DelayedBlock& block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0, "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask([this, block]() {
    std::ifstream file_stream(this->file_path_, std::ios::binary);
    file_stream.seekg(block.offset);
    file_stream.read(reinterpret_cast<char*>(block.data), block.size);
  });
  file_stream_.seekg(block.size, std::ios::cur);
}

void FileStream::WaitForDelayedBlock() {
  thread_pool_.Wait();
}

void FileStream::StartThreadPool(size_t n_threads) {
  thread_pool_.Start(n_threads);
}

//////////////////////
// TwoFilesWriteStream
//////////////////////

TwoFilesWriteStream::TwoFilesWriteStream(const std::string& file_path, const std::string& weights_file)
    : FileWriteStream(file_path), weights_stream_(weights_file) {}

void TwoFilesWriteStream::write_raw_bytes_in_second_stream(const uint8_t* ptr, offset_t n_bytes) {
  position_cache_[ptr] = weights_stream_.size();
  weights_stream_.write_raw_bytes(ptr, n_bytes);
}

TwoFilesStream::TwoFilesStream(const std::string& file_path, const std::string& weights_file)
    : FileStream(file_path), weights_stream_(weights_file) {}

void TwoFilesStream::read_bytes_from_weights_stream(offset_t n_bytes, uint8_t* pre_allocated_buffer, offset_t offset) {
  if (offset >= 0) {
    weights_stream_.file_stream_.seekg(offset);
  }
  weights_stream_.read_bytes(n_bytes, pre_allocated_buffer);
}

void TwoFilesStream::ReadDelayedBlock(DelayedBlock& block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0 || block.stream_id == 1,
      "Only two streams are allowed to read delayed blocks, but stream_id=",
      block.stream_id);
  blocks_.push_back(block);
  if (block.stream_id == 0) {
    thread_pool_.SubmitTask([this, block]() {
      std::ifstream file_stream(this->file_path(), std::ios::binary);
      file_stream.seekg(block.offset);
      file_stream.read(reinterpret_cast<char*>(block.data), block.size);
    });
    file_stream_.seekg(block.size, std::ios::cur);
  } else {
    thread_pool_.SubmitTask([this, block]() {
      std::ifstream file_stream(this->weights_file_path(), std::ios::binary);
      file_stream.seekg(block.offset);
      file_stream.read(reinterpret_cast<char*>(block.data), block.size);
    });
    weights_stream_.file_stream_.seekg(block.size, std::ios::cur);
  }
}

} // namespace utils
} // namespace onnx2
