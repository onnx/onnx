
#include "attr_proto_util.h"

namespace ONNX_NAMESPACE {

#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, ATTR_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttributeProto* out) {    \
    out->set_type(AttributeProto_AttributeType_##ATTR_TYPE);  \
    out->set_##FIELD(value);                                  \
  }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, ATTR_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttributeProto* out) {     \
    out->set_type(AttributeProto_AttributeType_##ATTR_TYPE);   \
    out->clear_##FIELD();                                      \
    for (const auto& v : value) {                              \
      out->add_##FIELD(v);                                     \
    }                                                          \
  }

DEFINE_SET_ATTR_VALUE_ONE(float, FLOAT, f);
DEFINE_SET_ATTR_VALUE_ONE(int64_t, INT, i);
DEFINE_SET_ATTR_VALUE_ONE(const std::string&, STRING, s);

DEFINE_SET_ATTR_VALUE_LIST(std::vector<float>, FLOATS, floats);
DEFINE_SET_ATTR_VALUE_LIST(std::vector<int64_t>, INTS, ints);
DEFINE_SET_ATTR_VALUE_LIST(const std::vector<std::string>&, STRINGS, strings);

void SetAttrValue(const AttributeProto& value, AttributeProto* out) {
  *out = value;
}

} // namespace ONNX_NAMESPACE
