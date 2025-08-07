#pragma once

#include <stdint.h>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common_helpers.h"
#include "simple_string.h"
#include "thread_pool.h"

namespace onnx2 {
namespace utils {

typedef int64_t offset_t;

inline int64_t decodeZigZag64(uint64_t n) {
  return (n >> 1) ^ -(n & 1);
}
inline uint64_t encodeZigZag64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

class StringStream;

struct FieldNumber {
  uint64_t field_number;
  uint64_t wire_type;
  std::string string() const;
};

struct DelayedBlock {
  uint64_t size;
  uint8_t* data;
  offset_t offset;
  uint8_t stream_id = 0; // this is used to identify the substream the data should be coming from
};

class BinaryStream {
 public:
  explicit inline BinaryStream() {}
  virtual ~BinaryStream();
  virtual bool ExternalWeights() const {
    return false;
  }
  // to overwrite
  virtual uint64_t next_uint64() = 0;
  virtual void CanRead(uint64_t len, const char* msg) = 0;
  virtual bool NotEnd() const = 0;
  virtual offset_t tell() const = 0;
  virtual std::string tell_around() const = 0;
  virtual const uint8_t* read_bytes(offset_t n_bytes, uint8_t* pre_allocated_buffer = nullptr) = 0;
  virtual void skip_bytes(offset_t n_bytes) = 0;
  virtual int64_t size() const = 0;
  // defines from the previous ones
  virtual RefString next_string();
  virtual int64_t next_int64();
  virtual int32_t next_int32();
  virtual float next_float();
  virtual double next_double();
  virtual FieldNumber next_field();
  template <typename T>
  void next_packed_element(T& value) {
    value = *reinterpret_cast<const T*>(read_bytes(sizeof(T)));
  }
  // Reading substream
  virtual void LimitToNext(uint64_t len);
  virtual void Restore();

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const {
    return false;
  }
  virtual void StartThreadPool(size_t n_threads);
  virtual void ReadDelayedBlock(DelayedBlock& block);
  virtual void WaitForDelayedBlock();

 protected:
  virtual void LimitTo(uint64_t len) = 0;
  virtual void _check();
  std::vector<uint64_t> limits_;
};

class StringWriteStream;
class BorrowedWriteStream;
class FileStream;

class BinaryWriteStream {
 public:
  explicit inline BinaryWriteStream() {}
  virtual ~BinaryWriteStream() {}
  // to overwrite
  virtual void write_raw_bytes(const uint8_t* data, offset_t n_bytes) = 0;
  virtual void write_raw_bytes_in_second_stream(const uint8_t* data, offset_t n_bytes);
  virtual int64_t size() const = 0;
  virtual const uint8_t* data() const = 0;
  // defined from the previous ones
  virtual void write_variant_uint64(uint64_t value);
  virtual void write_int64(int64_t value);
  virtual void write_int32(int32_t value);
  virtual void write_float(float value);
  virtual void write_double(double value);
  virtual void write_string(const std::string& value);
  virtual void write_string(const String& value);
  virtual void write_string(const RefString& value);
  virtual void write_string_stream(const StringWriteStream& stream);
  virtual void write_string_stream(const BorrowedWriteStream& stream);
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);
  template <typename T>
  void write_packed_element(const T& value) {
    write_raw_bytes(reinterpret_cast<const uint8_t*>(&value), sizeof(T));
  }
  // size
  virtual uint64_t size_field_header(uint32_t field_number, uint8_t wire_type);
  virtual uint64_t VarintSize(uint64_t value);
  virtual uint64_t size_variant_uint64(uint64_t value);
  virtual uint64_t size_int64(int64_t value);
  virtual uint64_t size_int32(int32_t value);
  virtual uint64_t size_float(float value);
  virtual uint64_t size_double(double value);
  virtual uint64_t size_string(const std::string& value);
  virtual uint64_t size_string(const String& value);
  virtual uint64_t size_string(const RefString& value);
  virtual uint64_t size_string_stream(const StringWriteStream& stream);
  virtual uint64_t size_string_stream(const BorrowedWriteStream& stream);
  // weights
  virtual bool ExternalWeights() const {
    return false;
  }

  // cache
  virtual void CacheSize(const void* ptr, uint64_t size);
  virtual bool GetCachedSize(const void* ptr, uint64_t& size);

 protected:
  std::unordered_map<const void*, uint64_t> size_cache_;
};

///////////
/// strings
///////////

class StringStream : public BinaryStream {
  friend class FileStream;

 public:
  explicit inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  explicit inline StringStream(const uint8_t* data, int64_t size) : BinaryStream(), pos_(0), size_(size), data_(data) {}
  void Setup(const uint8_t* data, int64_t size);
  virtual void CanRead(uint64_t len, const char* msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t* read_bytes(offset_t n_bytes, uint8_t* pre_allocated_buffer = nullptr) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  virtual bool NotEnd() const override {
    return pos_ < size_;
  }
  virtual offset_t tell() const override {
    return static_cast<offset_t>(pos_);
  }
  virtual std::string tell_around() const override;
  virtual inline int64_t size() const override {
    return size_;
  }

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const override {
    return thread_pool_.IsStarted();
  }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void ReadDelayedBlock(DelayedBlock& block) override;
  virtual void WaitForDelayedBlock() override;

 protected:
  virtual void LimitTo(uint64_t len) override;

 protected:
  offset_t pos_;
  offset_t size_;
  const uint8_t* data_;

 protected:
  // parallelization
  std::vector<DelayedBlock> blocks_;
  ThreadPool thread_pool_;
};

class StringWriteStream : public BinaryWriteStream {
 public:
  explicit inline StringWriteStream() : BinaryWriteStream(), buffer_() {}
  virtual void write_raw_bytes(const uint8_t* data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t* data() const override;

 protected:
 protected:
  std::vector<uint8_t> buffer_;
};

class BorrowedWriteStream : public BinaryWriteStream {
 public:
  explicit inline BorrowedWriteStream(const uint8_t* data, int64_t size)
      : BinaryWriteStream(), data_(data), size_(size) {}
  virtual void write_raw_bytes(const uint8_t* data, offset_t n_bytes) override;
  virtual int64_t size() const override {
    return size_;
  }
  virtual const uint8_t* data() const override {
    return data_;
  }

 protected:
  const uint8_t* data_;
  int64_t size_;
};

////////
// files
////////

class FileWriteStream : public BinaryWriteStream {
 public:
  explicit FileWriteStream(const std::string& file_path);
  virtual void write_raw_bytes(const uint8_t* data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t* data() const override;
  inline const std::string& file_path() const {
    return file_path_;
  }

 protected:
  std::string file_path_;
  std::ofstream file_stream_;
  uint64_t written_bytes_;
};

class TwoFilesStream;

class FileStream : public BinaryStream {
  friend class TwoFilesStream;

 public:
  explicit FileStream(const std::string& file_path);
  virtual ~FileStream();
  inline const std::string& file_path() const {
    return file_path_;
  }
  virtual void CanRead(uint64_t len, const char* msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t* read_bytes(offset_t n_bytes, uint8_t* pre_allocated_buffer = nullptr) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  /**
   * This is a dangerous zone. StreamStream points to the buffer_.data().
   * buffer_ changes everytime new bytes are read from the file.
   * So unlock() must be called or this class raises an exception.
   */
  virtual bool NotEnd() const override;
  virtual offset_t tell() const override;
  virtual std::string tell_around() const override;
  virtual bool is_open() const;
  virtual int64_t size() const override {
    return size_;
  }

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const override {
    return thread_pool_.IsStarted();
  }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void ReadDelayedBlock(DelayedBlock& block) override;
  virtual void WaitForDelayedBlock() override;

 protected:
  virtual void LimitTo(uint64_t len) override;

 protected:
  bool lock_;
  std::string file_path_;
  std::ifstream file_stream_;
  int64_t size_;
  std::vector<uint8_t> buffer_;
  // parallelization
  std::vector<DelayedBlock> blocks_;
  ThreadPool thread_pool_;
};

//////////////////////////////
// Stream for external weights
//////////////////////////////

class TwoFilesWriteStream : public FileWriteStream {
 public:
  explicit TwoFilesWriteStream(const std::string& file_path, const std::string& weights_file);
  inline const std::string& weights_file_path() const {
    return weights_stream_.file_path();
  }
  virtual bool ExternalWeights() const override {
    return true;
  }
  virtual void write_raw_bytes_in_second_stream(const uint8_t* data, offset_t n_bytes);
  virtual int64_t weights_size() const {
    return weights_stream_.size();
  }

 protected:
  FileWriteStream weights_stream_;
  std::unordered_map<const void*, uint64_t> position_cache_;
};

class TwoFilesStream : public FileStream {
 public:
  explicit TwoFilesStream(const std::string& file_path, const std::string& weights_file);
  inline const std::string& weights_file_path() const {
    return weights_stream_.file_path();
  }
  inline uint64_t weights_tell() const {
    return weights_stream_.tell();
  }
  virtual bool ExternalWeights() const override {
    return true;
  }
  virtual void
  read_bytes_from_weights_stream(offset_t n_bytes, uint8_t* pre_allocated_buffer = nullptr, offset_t offset = -1);
  virtual void ReadDelayedBlock(DelayedBlock& block) override;
  virtual int64_t weights_size() const {
    return weights_stream_.size();
  }

 protected:
  FileStream weights_stream_;
  std::unordered_map<const void*, uint64_t> position_cache_;
};

} // namespace utils
} // namespace onnx2
