#pragma once

#include "simple_string.h"
#include "stream.h"

#define FIELD_VARINT 0
// #define FIELD_FIXED64 1
#define FIELD_FIXED_SIZE 2
#define FIELD_FIXED32 5 // deprecated value but used in old files

#define SERIALIZATION_METHOD()                                                                         \
  uint64_t SerializeSize() const;                                                                      \
  void ParseFromString(const std::string &raw);                                                        \
  void ParseFromString(const std::string &raw, ParseOptions &opts);                                    \
  void SerializeToString(std::string &out) const;                                                      \
  void SerializeToString(std::string &out, SerializeOptions &opts) const;                              \
  uint64_t SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &opts) const;              \
  void ParseFromStream(utils::BinaryStream &stream, ParseOptions &options);                            \
  void SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const;           \
  std::vector<std::string> PrintToVectorString(utils::PrintOptions &options) const;

#define BEGIN_PROTO(cls, doc)                                                                          \
  class cls : public Message {                                                                         \
  public:                                                                                              \
    static inline constexpr const char *DOC = doc;                                                     \
    explicit inline cls() {}                                                                           \
    void CopyFrom(const cls &proto);

#define BEGIN_PROTO_NOINIT(cls, doc)                                                                   \
  class cls : public Message {                                                                         \
  public:                                                                                              \
    static inline constexpr const char *DOC = doc;                                                     \
    void CopyFrom(const cls &proto);

#define END_PROTO()                                                                                    \
  SERIALIZATION_METHOD()                                                                               \
  }                                                                                                    \
  ;

#if defined(FIELD)
#pragma error("macro FIELD is already defined.")
#endif

#define FIELD(type, name, order, doc)                                                                  \
public:                                                                                                \
  inline type &ref_##name() { return name##_; }                                                        \
  inline const type &ref_##name() const { return name##_; }                                            \
  inline const type *ptr_##name() const { return &name##_; }                                           \
  inline bool has_##name() const { return _has_field_(name##_); }                                      \
  inline void set_##name(const type &v) { name##_ = v; }                                               \
  inline int order_##name() const { return order; }                                                    \
  static inline constexpr const char *_name_##name = #name;                                            \
  static inline constexpr const char *DOC_##name = doc;                                                \
  type name##_;                                                                                        \
  using name##_t = type;

#define FIELD_DEFAULT(type, name, order, default_value, doc)                                           \
public:                                                                                                \
  inline type &ref_##name() { return name##_; }                                                        \
  inline const type &ref_##name() const { return name##_; }                                            \
  inline const type *ptr_##name() const { return &name##_; }                                           \
  inline bool has_##name() const { return _has_field_(name##_); }                                      \
  inline void set_##name(const type &v) { name##_ = v; }                                               \
  inline int order_##name() const { return order; }                                                    \
  static inline constexpr const char *_name_##name = #name;                                            \
  static inline constexpr const char *DOC_##name = doc;                                                \
  type name##_ = default_value;                                                                        \
  using name##_t = type;

#define FIELD_STR(name, order, doc)                                                                    \
  FIELD(utils::String, name, order, doc)                                                               \
  inline void set_##name(const std::string &v) { name##_ = v; }                                        \
  inline void set_##name(const utils::RefString &v) { name##_ = v; }

#define FIELD_REPEATED(type, name, order, doc)                                                         \
public:                                                                                                \
  inline utils::RepeatedField<type> &ref_##name() { return name##_; }                                  \
  inline const utils::RepeatedField<type> &ref_##name() const { return name##_; }                      \
  inline const utils::RepeatedField<type> *ptr_##name() const { return &name##_; }                     \
  inline type &add_##name() { return name##_.add(); }                                                  \
  inline type &add_##name(type &&v) {                                                                  \
    name##_.emplace_back(v);                                                                           \
    return name##_.back();                                                                             \
  }                                                                                                    \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }                  \
  inline int order_##name() const { return order; }                                                    \
  inline void clr_##name() { name##_.clear(); }                                                        \
  static inline constexpr const char *DOC_##name = doc;                                                \
  static inline constexpr const char *_name_##name = #name;                                            \
  inline bool packed_##name() const { return false; }                                                  \
  utils::RepeatedField<type> name##_;                                                                  \
  using name##_t = type;

#define FIELD_REPEATED_PROTO(type, name, order, doc)                                                   \
public:                                                                                                \
  inline utils::RepeatedProtoField<type> &ref_##name() { return name##_; }                             \
  inline const utils::RepeatedProtoField<type> &ref_##name() const { return name##_; }                 \
  inline const utils::RepeatedProtoField<type> *ptr_##name() const { return &name##_; }                \
  inline type &add_##name() { return name##_.add(); }                                                  \
  inline type &add_##name(const type &v) {                                                             \
    name##_.push_back(v);                                                                              \
    return name##_.back();                                                                             \
  }                                                                                                    \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }                  \
  inline int order_##name() const { return order; }                                                    \
  inline void clr_##name() { name##_.clear(); }                                                        \
  static inline constexpr const char *DOC_##name = doc;                                                \
  static inline constexpr const char *_name_##name = #name;                                            \
  inline bool packed_##name() const { return false; }                                                  \
  utils::RepeatedProtoField<type> name##_;                                                             \
  using name##_t = type;

#define FIELD_REPEATED_PACKED(type, name, order, doc)                                                  \
public:                                                                                                \
  inline utils::RepeatedField<type> &ref_##name() { return name##_; }                                  \
  inline const utils::RepeatedField<type> &ref_##name() const { return name##_; }                      \
  inline const utils::RepeatedField<type> *ptr_##name() const { return &name##_; }                     \
  inline type &add_##name() { return name##_.add(); }                                                  \
  inline type &add_##name(const type &v) {                                                             \
    name##_.push_back(v);                                                                              \
    return name##_.back();                                                                             \
  }                                                                                                    \
  inline bool has_##name() const { return _has_field_(name##_) && !name##_.empty(); }                  \
  inline int order_##name() const { return order; }                                                    \
  inline void clr_##name() { name##_.clear(); }                                                        \
  static inline constexpr const char *DOC_##name = doc;                                                \
  static inline constexpr const char *_name_##name = #name;                                            \
  inline bool packed_##name() const { return true; }                                                   \
  utils::RepeatedField<type> name##_;                                                                  \
  using name##_t = type;

#define _FIELD_OPTIONAL(type, name, order, doc)                                                        \
public:                                                                                                \
  inline type &ref_##name() {                                                                          \
    if (!has_##name()) {                                                                               \
      add_##name();                                                                                    \
    }                                                                                                  \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline const type &ref_##name() const {                                                              \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                    \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline const type *ptr_##name() const {                                                              \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                                  \
  }                                                                                                    \
  inline utils::OptionalField<type> &name##_optional() { return name##_; }                             \
  inline const utils::OptionalField<type> &name##_optional() const {                                   \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                    \
    return name##_;                                                                                    \
  }                                                                                                    \
  inline type &add_##name() {                                                                          \
    name##_.set_empty_value();                                                                         \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline void set_##name(const type &v) { name##_ = v; }                                               \
  inline void reset_##name() { name##_.reset(); }                                                      \
  inline bool has_##name() const { return name##_.has_value(); }                                       \
  inline int order_##name() const { return order; }                                                    \
  static inline constexpr const char *DOC_##name = doc;                                                \
  static inline constexpr const char *_name_##name = #name;                                            \
  utils::OptionalField<type> name##_;                                                                  \
  using name##_t = type;

#define FIELD_OPTIONAL(type, name, order, doc)                                                         \
  _FIELD_OPTIONAL(type, name, order, doc)                                                              \
  inline bool has_oneof_##name() const { return has_##name(); }

#define FIELD_OPTIONAL_ONEOF(type, name, order, oneof, doc)                                            \
  _FIELD_OPTIONAL(type, name, order, doc)                                                              \
  inline bool has_oneof_##name() const { return has_##oneof(); }

#define FIELD_OPTIONAL_ENUM(type, name, order, doc)                                                    \
public:                                                                                                \
  inline type &ref_##name() {                                                                          \
    if (!has_##name()) {                                                                               \
      add_##name();                                                                                    \
    }                                                                                                  \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline const type &ref_##name() const {                                                              \
    EXT_ENFORCE(name##_.has_value(), "Optional enum field '", #name, "' has no value.");               \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline const type *ptr_##name() const {                                                              \
    return has_##name() ? &(*name##_) : static_cast<type *>(nullptr);                                  \
  }                                                                                                    \
  inline utils::OptionalEnumField<type> &name##_optional() { return name##_; }                         \
  inline const utils::OptionalEnumField<type> &name##_optional() const {                               \
    EXT_ENFORCE(name##_.has_value(), "Optional field '", #name, "' has no value.");                    \
    return name##_;                                                                                    \
  }                                                                                                    \
  inline type &add_##name() {                                                                          \
    name##_.set_empty_value();                                                                         \
    return *name##_;                                                                                   \
  }                                                                                                    \
  inline void set_##name(const type &v) { name##_ = v; }                                               \
  inline void reset_##name() { name##_.reset(); }                                                      \
  inline bool has_##name() const { return name##_.has_value(); }                                       \
  inline int order_##name() const { return order; }                                                    \
  static inline constexpr const char *DOC_##name = doc;                                                \
  static inline constexpr const char *_name_##name = #name;                                            \
  utils::OptionalEnumField<type> name##_;                                                              \
  using name##_t = type;

namespace onnx2 {

struct ParseOptions {
  /** if true, raw data will not be read but skipped, tensors are not valid in that case  but the model
   * structure is still available */
  bool skip_raw_data = false;
  /** if skip_raw_data is true, raw data will be read only if it is larger than the threshold */
  int64_t raw_data_threshold = 1024;
  /** parallelizes the reading of the big blocks */
  bool parallel = false;
  /** number of threads to run in parallel if parallel is true, -1 for as many threads as the number of
   * cores */
  int32_t num_threads = -1;
};

struct SerializeOptions {
  /** if true, raw data will not be written but skipped, tensors are not valid in that case but the
   * model structure is still available */
  bool skip_raw_data = false;
  /** if skip_raw_data is true, raw data will be written only if it is larger than the threshold */
  int64_t raw_data_threshold = 1024;
};

using utils::offset_t;

template <typename T> inline bool _has_field_(const T &) { return true; }
template <> inline bool _has_field_(const utils::String &field) { return !field.empty(); }
template <> inline bool _has_field_(const std::vector<uint8_t> &field) { return !field.empty(); }

template <typename T> void CopyProtoFrom(T &dest, const T &src);

class Message {
public:
  explicit inline Message() {}
  inline bool operator==(const Message &) const {
    EXT_THROW("operator == not implemented for a Message");
  }
};

} // namespace onnx2