#include "onnx2.h"
#include "onnx2_helper.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace onnx2;

#define PYDEFINE_PROTO(m, cls)                                                                         \
  py::class_<cls, Message> py_##cls(m, #cls, cls::DOC);                                                \
  py_##cls.def(py::init<>())

#define PYDEFINE_SUBPROTO(m, cls, subname)                                                             \
  py::class_<cls::subname, Message> py_sub_##cls##subname(m, #subname, cls::subname::DOC);             \
  py_sub_##cls##subname.def(py::init<>())

#define PYDEFINE_PROTO_WITH_SUBTYPES(m, cls)                                                           \
  py::class_<cls, Message> py_##cls(m, #cls, cls::DOC);                                                \
  py_##cls.def(py::init<>());

#define _PYADD_PROTO_SERIALIZATION(cls, name_inst) pyadd_proto_serialization(name_inst);

#define PYADD_PROTO_SERIALIZATION(cls) _PYADD_PROTO_SERIALIZATION(cls, py_##cls)
#define PYADD_SUBPROTO_SERIALIZATION(cls, sub) _PYADD_PROTO_SERIALIZATION(cls::sub, py_sub_##cls##sub)

#define PYFIELD(cls, name)                                                                             \
  def_readwrite(#name, &cls::name##_, cls::DOC_##name)                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_STR(cls, name)                                                                         \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](const cls &self) -> std::string {                                                             \
        std::string s = self.ref_##name().as_string();                                                 \
        return s;                                                                                      \
      },                                                                                               \
      [](cls &self, py::object obj) {                                                                  \
        if (py::isinstance<py::str>(obj)) {                                                            \
          std::string st = obj.cast<std::string>();                                                    \
          self.set_##name(st);                                                                         \
        } else if (py::isinstance<py::bytes>(obj)) {                                                   \
          std::string st = obj.cast<py::bytes>();                                                      \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(obj.cast<cls::name##_t &>());                                                \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define PYFIELD_STR_AS_BYTES(cls, name)                                                                \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](const cls &self) -> py::bytes {                                                               \
        std::string s = py::bytes(self.ref_##name().as_string());                                      \
        return s;                                                                                      \
      },                                                                                               \
      [](cls &self, py::object obj) {                                                                  \
        if (py::isinstance<py::str>(obj)) {                                                            \
          std::string st = obj.cast<std::string>();                                                    \
          self.set_##name(st);                                                                         \
        } else if (py::isinstance<py::bytes>(obj)) {                                                   \
          std::string st = obj.cast<py::bytes>();                                                      \
          self.set_##name(st);                                                                         \
        } else {                                                                                       \
          self.set_##name(obj.cast<cls::name##_t &>());                                                \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value")

#define _PYFIELD_OPTIONAL_CTYPE(cls, name, ctype)                                                      \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](cls &self) -> py::object {                                                                    \
        if (!self.has_##name())                                                                        \
          return py::none();                                                                           \
        return py::cast(self.ref_##name(), py::return_value_policy::reference);                        \
      },                                                                                               \
      [](cls &self, py::object obj) {                                                                  \
        if (obj.is_none()) {                                                                           \
          self.reset_##name();                                                                         \
        } else if (py::isinstance<py::ctype##_>(obj)) {                                                \
          self.set_##name(obj.cast<ctype>());                                                          \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")

#define PYFIELD_OPTIONAL_INT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, int)
#define PYFIELD_OPTIONAL_FLOAT(cls, name) _PYFIELD_OPTIONAL_CTYPE(cls, name, float)

#define PYFIELD_OPTIONAL_PROTO(cls, name)                                                              \
  def_property(                                                                                        \
      #name,                                                                                           \
      [](cls &self) -> py::object {                                                                    \
        if (!self.name##_.has_value()) {                                                               \
          if (self.has_oneof_##name())                                                                 \
            return py::none();                                                                         \
          self.name##_.set_empty_value();                                                              \
        }                                                                                              \
        return py::cast(*self.name##_, py::return_value_policy::reference);                            \
      },                                                                                               \
      [](cls &self, py::object obj) {                                                                  \
        if (obj.is_none()) {                                                                           \
          self.name##_.reset();                                                                        \
        } else if (py::isinstance<cls::name##_t>(obj)) {                                               \
          self.name##_ = obj.cast<cls::name##_t &>();                                                  \
        } else {                                                                                       \
          EXT_THROW("unexpected value type, unable to set '" #name "' for class '" #cls "'.");         \
        }                                                                                              \
      },                                                                                               \
      cls::DOC_##name)                                                                                 \
      .def("has_" #name, &cls::has_##name, "Tells if '" #name "' has a value.")                        \
      .def(                                                                                            \
          "add_" #name, [](cls & self) -> cls::name##_t & {                                            \
            self.name##_.set_empty_value();                                                            \
            return *self.name##_;                                                                      \
          },                                                                                           \
          py::return_value_policy::reference, "Sets an empty value.")

#define SHORTEN_CODE(cls, dtype)                                                                       \
  def_property_readonly_static(#dtype, [](py::object) -> int { return static_cast<int>(cls::dtype); })

#define DECLARE_REPEATED_FIELD(T, inst_name)                                                           \
  py::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);

#define DECLARE_REPEATED_FIELD_PROTO(T, inst_name)                                                     \
  py::class_<utils::RepeatedField<T>> inst_name(m, "RepeatedField" #T, "RepeatedField" #T);            \
  py::class_<utils::RepeatedProtoField<T>> inst_name##_proto(m, "RepeatedProtoField" #T,               \
                                                             "RepeatedProtoField" #T);

#define DECLARE_REPEATED_FIELD_SUBPROTO(cls, T, inst_name)                                             \
  py::class_<utils::RepeatedField<cls::T>> inst_name(m, "RepeatedField" #cls #T,                       \
                                                     "RepeatedField" #cls #T);                         \
  py::class_<utils::RepeatedProtoField<cls::T>> inst_name##_proto(m, "RepeatedProtoField" #cls #T,     \
                                                                  "RepeatedProtoField" #cls #T);

template <typename cls> void pyadd_proto_serialization(py::class_<cls, Message> &name_inst) {
  name_inst
      .def(
          "ParseFromString",
          [](cls &self, py::bytes data, py::object options) {
            std::string raw = data.cast<std::string>();
            if (py::isinstance<ParseOptions &>(options)) {
              self.ParseFromString(raw, options.cast<ParseOptions &>());
            } else {
              self.ParseFromString(raw);
            }
          },
          py::arg("data"), py::arg("options") = py::none(),
          "Parses a sequence of bytes to fill this instance.")
      .def(
          "ParseFromFile",
          [](cls &self, const std::string &file_path, py::object options,
             const std::string &external_data_file) {
            utils::FileStream *stream = external_data_file.empty()
                                            ? new utils::FileStream(file_path)
                                            : new utils::TwoFilesStream(file_path, external_data_file);
            if (py::isinstance<ParseOptions &>(options)) {
              ParseOptions &coptions = options.cast<ParseOptions &>();
              if (coptions.parallel) {
                stream->StartThreadPool(coptions.num_threads);
              }
              ParseProtoFromStream(self, *stream, coptions);
              if (coptions.parallel) {
                stream->WaitForDelayedBlock();
              }
            } else {
              ParseOptions opts;
              ParseProtoFromStream(self, *stream, opts);
            }
            delete stream;
          },
          py::arg("name"), py::arg("options") = py::none(), py::arg("external_data_file") = "",
          "Parses a binary file to fill this instance.")
      .def(
          "SerializeSize",
          [](cls &self, py::object options) -> uint64_t {
            if (py::isinstance<SerializeOptions &>(options)) {
              utils::StringWriteStream out;
              return self.SerializeSize(out, options.cast<SerializeOptions &>());
            } else {
              return self.SerializeSize();
            }
          },
          py::arg("options") = py::none(), "Returns the size once serialized without serializing.")
      .def(
          "SerializeToString",
          [](cls &self, py::object options) {
            std::string out;
            if (py::isinstance<SerializeOptions &>(options)) {
              self.SerializeToString(out, options.cast<SerializeOptions &>());
            } else {
              SerializeOptions opts;
              self.SerializeToString(out, opts);
            }
            return py::bytes(out);
          },
          py::arg("options") = py::none(), "Serializes this instance into a sequence of bytes.")
      .def(
          "SerializeToFile",
          [](cls &self, const std::string &file_path, py::object options,
             std::string &external_data_file) {
            utils::BinaryWriteStream *stream =
                external_data_file.empty()
                    ? new utils::FileWriteStream(file_path)
                    : new utils::TwoFilesWriteStream(file_path, external_data_file);
            if (py::isinstance<SerializeOptions &>(options)) {
              SerializeProtoToStream(self, *stream, options.cast<SerializeOptions &>(),
                                     !external_data_file.empty());
            } else {
              SerializeOptions opts;
              SerializeProtoToStream(self, *stream, opts, !external_data_file.empty());
            }
            delete stream;
          },
          py::arg("name"), py::arg("options") = py::none(), py::arg("external_data_file") = "",
          "Serializes this instance into a file. If ``external_data_size`` is not empty, big weights "
          "are stored in this (depending on ``options.raw_data_threshold``.")
      .def(
          "__str__",
          [](cls &self) -> std::string {
            utils::PrintOptions opts;
            std::vector<std::string> rows = self.PrintToVectorString(opts);
            return utils::join_string(rows);
          },
          "Creates a printable string for this class.")
      .def(
          "CopyFrom", [](cls &self, const cls &src) { self.CopyFrom(src); },
          "Copies one instance into this one.")
      .def(
          "__eq__",
          [](const cls &self, const cls &other) -> bool {
            SerializeOptions opts1, opts2;
            std::string s1;
            self.SerializeToString(s1, opts1);
            std::string s2;
            other.SerializeToString(s2, opts2);
            return s1 == s2;
          },
          py::arg("other"), "Compares the serialized strings.");
}

template <typename T> void define_repeated_field_type(py::class_<utils::RepeatedField<T>> &pycls) {
  pycls.def(py::init<>())
      .def("add", &utils::RepeatedField<T>::add, py::return_value_policy::reference,
           "Adds an empty element.")
      .def("clear", &utils::RepeatedField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedField<T>::size, "Returns the number of elements.")
      .def(
          "__getitem__",
          [](utils::RepeatedField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self[index];
          },
          py::return_value_policy::reference, py::arg("index"),
          "Returns the element at position index.")
      .def(
          "__delitem__",
          [](utils::RepeatedField<T> &self, py::slice slice) {
            size_t start, stop, step, slicelength;
            if (slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
              self.remove_range(start, stop, step);
            }
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedField<T> &self) { return py::make_iterator(self.begin(), self.end()); },
          py::keep_alive<0, 1>(), "Iterates over the elements.");
}

template <typename T>
void define_repeated_field_type_extend(py::class_<utils::RepeatedField<T>> &pycls) {
  pycls
      .def(
          "append", [](utils::RepeatedField<T> &self, T v) { self.push_back(v); }, py::arg("item"),
          "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, py::iterable iterable) {
            if (py::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(iterable.cast<utils::RepeatedField<T> &>());
            } else {
              self.extend(iterable.cast<std::vector<T>>());
            }
          },
          py::arg("sequence"), "Extends the list of values.");
}

template <>
void define_repeated_field_type_extend(py::class_<utils::RepeatedField<utils::String>> &pycls) {
  pycls
      .def(
          "append",
          [](utils::RepeatedField<utils::String> &self, const utils::String &v) { self.push_back(v); },
          py::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<utils::String> &self, py::iterable iterable) {
            if (py::isinstance<utils::RepeatedField<utils::String>>(iterable)) {
              self.extend(iterable.cast<utils::RepeatedField<utils::String> &>());
            } else {
              std::vector<utils::String> values;
              for (auto it : iterable) {
                if (py::isinstance<utils::String>(it)) {
                  values.push_back(it.cast<utils::String &>());
                } else {
                  values.emplace_back(utils::String(it.cast<std::string>()));
                }
              }
              self.extend(values);
            }
          },
          py::arg("sequence"), "Extends the list of values.");
}

template <typename T>
void define_repeated_field_type_proto(py::class_<utils::RepeatedField<T>> &pycls,
                                      py::class_<utils::RepeatedProtoField<T>> &pycls_proto) {
  define_repeated_field_type(pycls);
  pycls
      .def(
          "append", [](utils::RepeatedField<T> &self, const T &v) { self.push_back(v); },
          py::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedField<T> &self, py::iterable iterable) {
            if (py::isinstance<utils::RepeatedField<T>>(iterable)) {
              self.extend(iterable.cast<utils::RepeatedField<T> &>());
            } else {
              py::list els = iterable.cast<py::list>();
              for (auto it : els) {
                if (py::isinstance<const T &>(it)) {
                  self.push_back(it.cast<T>());
                } else if (py::isinstance<T>(it)) {
                  self.push_back(it.cast<T>());
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          py::arg("sequence"), "Extends the list of values.");
  pycls_proto.def(py::init<>())
      .def("add", &utils::RepeatedProtoField<T>::add, py::return_value_policy::reference,
           "Adds an empty element.")
      .def("clear", &utils::RepeatedProtoField<T>::clear, "Removes every element.")
      .def("__len__", &utils::RepeatedProtoField<T>::size, "Returns the number of elements.")
      .def(
          "__getitem__",
          [](utils::RepeatedProtoField<T> &self, int index) -> T & {
            if (index < 0)
              index += static_cast<int>(self.size());
            EXT_ENFORCE(index >= 0 && index < static_cast<int>(self.size()), "index=", index,
                        " out of boundary");
            return self[index];
          },
          py::return_value_policy::reference, py::arg("index"),
          "Returns the element at position index.")
      .def(
          "__delitem__",
          [](utils::RepeatedProtoField<T> &self, py::slice slice) {
            size_t start, stop, step, slicelength;
            if (slice.compute(self.size(), &start, &stop, &step, &slicelength)) {
              self.remove_range(start, stop, step);
            }
          },
          "Removes elements.")
      .def(
          "__iter__",
          [](utils::RepeatedProtoField<T> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(), "Iterates over the elements.")
      .def(
          "__eq__",
          [](utils::RepeatedField<T> &self, py::list &obj) -> bool {
            if (self.size() != obj.size())
              return false;
            for (size_t i = 0; i < self.size(); ++i) {
              if (!py::isinstance<T &>(obj[i]))
                return false;
              std::string s1, s2;
              self[i].SerializeToString(s1);
              obj[i].cast<T &>().SerializeToString(s2);
              if (s1 != s2)
                return false;
            }
            return true;
          },
          "Compares the container to a list of objects.")
      .def(
          "append", [](utils::RepeatedProtoField<T> &self, const T &v) { self.push_back(v); },
          py::arg("item"), "Append one element to the list of values.")
      .def(
          "extend",
          [](utils::RepeatedProtoField<T> &self, py::iterable iterable) {
            if (py::isinstance<utils::RepeatedProtoField<T>>(iterable)) {
              self.extend(iterable.cast<utils::RepeatedProtoField<T> &>());
            } else {
              py::list els = iterable.cast<py::list>();
              for (auto it : els) {
                if (py::isinstance<const T &>(it)) {
                  self.push_back(it.cast<const T &>());
                } else if (py::isinstance<T>(it)) {
                  self.push_back(it.cast<T>());
                } else {
                  EXT_THROW("Unable to cast an element of type into ", typeid(T).name());
                }
              }
            }
          },
          py::arg("sequence"), "Extends the list of values.");
}

PYBIND11_MODULE(_onnx2py, m) {
  m.doc() =
#if defined(__APPLE__)
      "onnx from python without protobuf but using the same format"
#else
      R"pbdoc(onnx from python without protobuf but using the same format)pbdoc"
#endif
      ;

  m.def(
      "utils_onnx2_read_varint64",
      [](py::bytes data) -> py::tuple {
        std::string raw = data;
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
        utils::StringStream st(ptr, raw.size());
        int64_t value = st.next_int64();
        return py::make_tuple(value, st.tell());
      },
      py::arg("data"),
      R"pbdoc(Reads a int64_t (protobuf format)
:param data: bytes
:return: 2-tuple, value and number of read bytes
)pbdoc");

  py::class_<ParseOptions>(m, "ParseOptions", "Parsing options for proto classes")
      .def(py::init<>())
      .def_readwrite("skip_raw_data", &ParseOptions::skip_raw_data,
                     "if true, raw data will not be read but skipped, tensors are not valid in that "
                     "case  but the model structure is still available")
      .def_readwrite(
          "raw_data_threshold", &ParseOptions::raw_data_threshold,
          "if skip_raw_data is true, raw data will be read only if it is larger than the threshold")
      .def_readwrite("parallel", &ParseOptions::parallel, "parallelizes the reading of the big blocks")
      .def_readwrite("num_threads", &ParseOptions::num_threads,
                     "number of threads to run in parallel if parallel is true, -1 for as many threads "
                     "as the number of cores");

  py::class_<SerializeOptions>(m, "SerializeOptions", "Serializing options for proto classes")
      .def(py::init<>())
      .def_readwrite("skip_raw_data", &SerializeOptions::skip_raw_data,
                     "if true, raw data will not be written but skipped, tensors are not valid in that "
                     "case  but the model structure is still available")
      .def_readwrite(
          "raw_data_threshold", &SerializeOptions::raw_data_threshold,
          "if skip_raw_data is true, raw data will be written only if it is larger than the threshold");

  py::class_<utils::PrintOptions>(m, "PrintOptions", "Printing options for proto classes")
      .def(py::init<>())
      .def_readwrite("skip_raw_data", &utils::PrintOptions::skip_raw_data,
                     "if true, raw data will not be printed but skipped, tensors are not valid in that "
                     "case  but the model structure is still available")
      .def_readwrite(
          "raw_data_threshold", &utils::PrintOptions::raw_data_threshold,
          "if skip_raw_data is true, raw data will be printed only if it is larger than the threshold");

  py::class_<utils::String>(m, "String", "Simplified string with no final null character.")
      .def(py::init<std::string>())
      .def(
          "__str__", [](const utils::String &self) -> std::string { return self.as_string(); },
          "Converts this instance into a python string.")
      .def(
          "__repr__",
          [](const utils::String &self) -> std::string {
            return std::string("'") + self.as_string() + std::string("'");
          },
          "Represention with surrounding quotes.")
      .def(
          "__len__", [](const utils::String &self) -> int { return self.size(); },
          "Returns the length of the string.")
      .def(
          "__eq__", [](const utils::String &self, const std::string &s) -> int { return self == s; },
          "Compares two strings.");

  DECLARE_REPEATED_FIELD(int64_t, rep_int64_t);
  define_repeated_field_type(rep_int64_t);
  define_repeated_field_type_extend(rep_int64_t);

  DECLARE_REPEATED_FIELD(int32_t, rep_int32_t);
  define_repeated_field_type(rep_int32_t);
  define_repeated_field_type_extend(rep_int32_t);

  DECLARE_REPEATED_FIELD(uint64_t, rep_uint64_t);
  define_repeated_field_type(rep_uint64_t);
  define_repeated_field_type_extend(rep_uint64_t);

  DECLARE_REPEATED_FIELD(float, rep_float);
  define_repeated_field_type(rep_float);
  define_repeated_field_type_extend(rep_float);

  DECLARE_REPEATED_FIELD(double, rep_double);
  define_repeated_field_type(rep_double);
  define_repeated_field_type_extend(rep_double);

  py::class_<utils::RepeatedField<utils::String>> rep_string(m, "RepeatedFieldString",
                                                             "RepeatedFieldString");
  define_repeated_field_type(rep_string);
  define_repeated_field_type_extend(rep_string);

  py::enum_<OperatorStatus>(m, "OperatorStatus", py::arithmetic())
      .value("EXPERIMENTAL", OperatorStatus::EXPERIMENTAL)
      .value("STABLE", OperatorStatus::STABLE)
      .export_values();

  py::class_<Message>(m, "Message", "Message, base class for all onnx2 classes").def(py::init<>());

  PYDEFINE_PROTO(m, StringStringEntryProto)
      .PYFIELD_STR(StringStringEntryProto, key)
      .PYFIELD_STR(StringStringEntryProto, value);
  PYADD_PROTO_SERIALIZATION(StringStringEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(StringStringEntryProto, rep_ssentry);
  define_repeated_field_type_proto(rep_ssentry, rep_ssentry_proto);

  PYDEFINE_PROTO(m, OperatorSetIdProto)
      .PYFIELD_STR(OperatorSetIdProto, domain)
      .PYFIELD(OperatorSetIdProto, version);
  PYADD_PROTO_SERIALIZATION(OperatorSetIdProto);
  DECLARE_REPEATED_FIELD_PROTO(OperatorSetIdProto, rep_osp);
  define_repeated_field_type_proto(rep_osp, rep_osp_proto);

  PYDEFINE_PROTO(m, TensorAnnotation)
      .PYFIELD_STR(TensorAnnotation, tensor_name)
      .PYFIELD(TensorAnnotation, quant_parameter_tensor_names);
  PYADD_PROTO_SERIALIZATION(TensorAnnotation);

  PYDEFINE_PROTO(m, IntIntListEntryProto)
      .PYFIELD(IntIntListEntryProto, key)
      .PYFIELD(IntIntListEntryProto, value);
  PYADD_PROTO_SERIALIZATION(IntIntListEntryProto);
  DECLARE_REPEATED_FIELD_PROTO(IntIntListEntryProto, rep_iil);
  define_repeated_field_type_proto(rep_iil, rep_iil_proto);

  PYDEFINE_PROTO(m, DeviceConfigurationProto)
      .PYFIELD_STR(DeviceConfigurationProto, name)
      .PYFIELD(DeviceConfigurationProto, num_devices)
      .PYFIELD(DeviceConfigurationProto, device);
  PYADD_PROTO_SERIALIZATION(DeviceConfigurationProto);

  PYDEFINE_PROTO(m, SimpleShardedDimProto)
      .PYFIELD_OPTIONAL_INT(SimpleShardedDimProto, dim_value)
      .PYFIELD_STR(SimpleShardedDimProto, dim_param)
      .PYFIELD(SimpleShardedDimProto, num_shards);
  PYADD_PROTO_SERIALIZATION(SimpleShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(SimpleShardedDimProto, rep_ssdp);
  define_repeated_field_type_proto(rep_ssdp, rep_ssdp_proto);

  PYDEFINE_PROTO(m, ShardedDimProto)
      .PYFIELD(ShardedDimProto, axis)
      .PYFIELD(ShardedDimProto, simple_sharding);
  PYADD_PROTO_SERIALIZATION(ShardedDimProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardedDimProto, rep_sdp);
  define_repeated_field_type_proto(rep_sdp, rep_sdp_proto);

  PYDEFINE_PROTO(m, ShardingSpecProto)
      .PYFIELD_STR(ShardingSpecProto, tensor_name)
      .PYFIELD(ShardingSpecProto, device)
      .PYFIELD(ShardingSpecProto, index_to_device_group_map)
      .PYFIELD(ShardingSpecProto, sharded_dim);
  PYADD_PROTO_SERIALIZATION(ShardingSpecProto);
  DECLARE_REPEATED_FIELD_PROTO(ShardingSpecProto, rep_ssp);
  define_repeated_field_type_proto(rep_ssp, rep_ssp_proto);

  PYDEFINE_PROTO(m, NodeDeviceConfigurationProto)
      .PYFIELD_STR(NodeDeviceConfigurationProto, configuration_id)
      .PYFIELD(NodeDeviceConfigurationProto, sharding_spec)
      .PYFIELD_OPTIONAL_INT(NodeDeviceConfigurationProto, pipeline_stage);
  PYADD_PROTO_SERIALIZATION(NodeDeviceConfigurationProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorShapeProto);
  PYDEFINE_SUBPROTO(py_TensorShapeProto, TensorShapeProto, Dimension)
      .PYFIELD_OPTIONAL_INT(TensorShapeProto::Dimension, dim_value)
      .PYFIELD_STR(TensorShapeProto::Dimension, dim_param)
      .PYFIELD_STR(TensorShapeProto::Dimension, denotation);
  PYADD_SUBPROTO_SERIALIZATION(TensorShapeProto, Dimension);
  DECLARE_REPEATED_FIELD_SUBPROTO(TensorShapeProto, Dimension, rep_tspd);
  define_repeated_field_type_proto(rep_tspd, rep_tspd_proto);
  py_TensorShapeProto.PYFIELD(TensorShapeProto, dim);
  PYADD_PROTO_SERIALIZATION(TensorShapeProto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TensorProto);

  py::enum_<TensorProto::DataType>(py_TensorProto, "DataType", py::arithmetic())
      .value("UNDEFINED", TensorProto::DataType::UNDEFINED)
      .value("FLOAT", TensorProto::DataType::FLOAT)
      .value("UINT8", TensorProto::DataType::UINT8)
      .value("INT8", TensorProto::DataType::INT8)
      .value("UINT16", TensorProto::DataType::UINT16)
      .value("INT16", TensorProto::DataType::INT16)
      .value("INT32", TensorProto::DataType::INT32)
      .value("INT64", TensorProto::DataType::INT64)
      .value("STRING", TensorProto::DataType::STRING)
      .value("BOOL", TensorProto::DataType::BOOL)
      .value("FLOAT16", TensorProto::DataType::FLOAT16)
      .value("DOUBLE", TensorProto::DataType::DOUBLE)
      .value("UINT32", TensorProto::DataType::UINT32)
      .value("UINT64", TensorProto::DataType::UINT64)
      .value("COMPLEX64", TensorProto::DataType::COMPLEX64)
      .value("COMPLEX128", TensorProto::DataType::COMPLEX128)
      .value("BFLOAT16", TensorProto::DataType::BFLOAT16)
      .value("FLOAT8E4M3FN", TensorProto::DataType::FLOAT8E4M3FN)
      .value("FLOAT8E4M3FNUZ", TensorProto::DataType::FLOAT8E4M3FNUZ)
      .value("FLOAT8E5M2", TensorProto::DataType::FLOAT8E5M2)
      .value("FLOAT8E5M2FNUZ", TensorProto::DataType::FLOAT8E5M2FNUZ)
      .value("UINT4", TensorProto::DataType::UINT4)
      .value("INT4", TensorProto::DataType::INT4)
      .value("FLOAT4E2M1", TensorProto::DataType::FLOAT4E2M1)
      .value("FLOAT8E8M0", TensorProto::DataType::FLOAT8E8M0)
      .export_values();
  py::enum_<TensorProto::DataLocation>(py_TensorProto, "DataLocation", py::arithmetic())
      .value("DEFAULT", TensorProto::DataLocation::DEFAULT)
      .value("EXTERNAL", TensorProto::DataLocation::EXTERNAL)
      .export_values();
  py_TensorProto.SHORTEN_CODE(TensorProto::DataType, UNDEFINED)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT)
      .SHORTEN_CODE(TensorProto::DataType, UINT8)
      .SHORTEN_CODE(TensorProto::DataType, INT8)
      .SHORTEN_CODE(TensorProto::DataType, UINT16)
      .SHORTEN_CODE(TensorProto::DataType, INT16)
      .SHORTEN_CODE(TensorProto::DataType, INT32)
      .SHORTEN_CODE(TensorProto::DataType, INT64)
      .SHORTEN_CODE(TensorProto::DataType, STRING)
      .SHORTEN_CODE(TensorProto::DataType, BOOL)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, DOUBLE)
      .SHORTEN_CODE(TensorProto::DataType, UINT32)
      .SHORTEN_CODE(TensorProto::DataType, UINT64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX64)
      .SHORTEN_CODE(TensorProto::DataType, COMPLEX128)
      .SHORTEN_CODE(TensorProto::DataType, BFLOAT16)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FN)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E4M3FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E5M2FNUZ)
      .SHORTEN_CODE(TensorProto::DataType, UINT4)
      .SHORTEN_CODE(TensorProto::DataType, INT4)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT4E2M1)
      .SHORTEN_CODE(TensorProto::DataType, FLOAT8E8M0)
      .PYFIELD(TensorProto, dims)
      .def_property(
          "data_type", [](const TensorProto &self) -> TensorProto::DataType { return self.data_type_; },
          [](TensorProto &self, py::object obj) {
            if (py::isinstance<py::int_>(obj)) {
              self.data_type_ = static_cast<TensorProto::DataType>(obj.cast<int>());
            } else {
              self.data_type_ = obj.cast<TensorProto::DataType>();
            }
          },
          TensorProto::DOC_data_type)
      .def_property(
          "data_location",
          [](const TensorProto &self) -> TensorProto::DataLocation {
            return self.has_data_location() ? *self.data_location_ : TensorProto::DataLocation::DEFAULT;
          },
          [](TensorProto &self, py::object obj) {
            if (py::isinstance<py::int_>(obj)) {
              self.data_location_ = static_cast<TensorProto::DataLocation>(obj.cast<int>());
            } else {
              self.data_location_ = obj.cast<TensorProto::DataLocation>();
            }
          },
          TensorProto::DOC_data_location)
      .PYFIELD_STR(TensorProto, name)
      .PYFIELD_STR(TensorProto, doc_string)
      .PYFIELD(TensorProto, external_data)
      .PYFIELD(TensorProto, metadata_props)
      .PYFIELD(TensorProto, dims)
      .PYFIELD(TensorProto, double_data)
      .PYFIELD(TensorProto, float_data)
      .PYFIELD(TensorProto, int64_data)
      .PYFIELD(TensorProto, int32_data)
      .PYFIELD(TensorProto, uint64_data)
      .def_property(
          "string_data",
          [](const TensorProto &self) -> py::list {
            py::list result;
            for (const auto &s : self.string_data_) {
              result.append(py::bytes(std::string(s.data(), s.size())));
            }
            return result;
          },
          [](TensorProto &self, py::list data) {
            self.string_data_.reserve(py::len(data));

            for (const auto &item : data) {
              if (py::isinstance<py::bytes>(item)) {
                self.string_data_.emplace_back(item.cast<std::string>());
              } else if (py::isinstance<py::str>(item)) {
                self.string_data_.emplace_back(item.cast<std::string>());
              } else {
                EXT_THROW("unable to convert one item from the list into a string")
              }
            }
          },
          TensorProto::DOC_string_data)
      .def_property(
          "raw_data",
          [](const TensorProto &self) -> py::bytes {
            return py::bytes(reinterpret_cast<const char *>(self.raw_data_.data()),
                             self.raw_data_.size());
          },
          [](TensorProto &self, py::bytes data) {
            std::string raw = data;
            const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
            self.raw_data_.resize(raw.size());
            memcpy(self.raw_data_.data(), ptr, raw.size());
          },
          TensorProto::DOC_raw_data);
  PYADD_PROTO_SERIALIZATION(TensorProto);
  DECLARE_REPEATED_FIELD_PROTO(TensorProto, rep_tp);
  define_repeated_field_type_proto(rep_tp, rep_tp_proto);

  PYDEFINE_PROTO(m, SparseTensorProto)
      .PYFIELD(SparseTensorProto, values)
      .PYFIELD(SparseTensorProto, indices)
      .PYFIELD(SparseTensorProto, dims);
  PYADD_PROTO_SERIALIZATION(SparseTensorProto);
  DECLARE_REPEATED_FIELD_PROTO(SparseTensorProto, rep_tsp);
  define_repeated_field_type_proto(rep_tsp, rep_tsp_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, TypeProto);
  PYDEFINE_SUBPROTO(py_TypeProto, TypeProto, Tensor)
      .PYFIELD_OPTIONAL_INT(TypeProto::Tensor, elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Tensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Tensor);
  PYDEFINE_SUBPROTO(py_TypeProto, TypeProto, SparseTensor)
      .PYFIELD_OPTIONAL_INT(TypeProto::SparseTensor, elem_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::SparseTensor, shape);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, SparseTensor);
  PYDEFINE_SUBPROTO(py_TypeProto, TypeProto, Sequence)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Sequence, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Sequence);
  PYDEFINE_SUBPROTO(py_TypeProto, TypeProto, Optional)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Optional, elem_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Optional);
  PYDEFINE_SUBPROTO(py_TypeProto, TypeProto, Map)
      .PYFIELD(TypeProto::Map, key_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto::Map, value_type);
  PYADD_SUBPROTO_SERIALIZATION(TypeProto, Map);
  py_TypeProto.PYFIELD_OPTIONAL_PROTO(TypeProto, tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sequence_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, map_type)
      .PYFIELD_STR(TypeProto, denotation)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, sparse_tensor_type)
      .PYFIELD_OPTIONAL_PROTO(TypeProto, optional_type);
  PYADD_PROTO_SERIALIZATION(TypeProto);

  PYDEFINE_PROTO(m, ValueInfoProto)
      .PYFIELD_STR(ValueInfoProto, name)
      .PYFIELD_OPTIONAL_PROTO(ValueInfoProto, type)
      .PYFIELD_STR(ValueInfoProto, doc_string)
      .PYFIELD(ValueInfoProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(ValueInfoProto);
  DECLARE_REPEATED_FIELD_PROTO(ValueInfoProto, rep_vip);
  define_repeated_field_type_proto(rep_vip, rep_vip_proto);

  PYDEFINE_PROTO_WITH_SUBTYPES(m, AttributeProto);
  py::enum_<AttributeProto::AttributeType> attribute_type(py_AttributeProto, "AttributeType",
                                                          py::arithmetic());
  attribute_type.value("UNDEFINED", AttributeProto::AttributeType::UNDEFINED)
      .value("FLOAT", AttributeProto::AttributeType::FLOAT)
      .value("INT", AttributeProto::AttributeType::INT)
      .value("STRING", AttributeProto::AttributeType::STRING)
      .value("GRAPH", AttributeProto::AttributeType::GRAPH)
      .value("SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR)
      .value("FLOATS", AttributeProto::AttributeType::FLOATS)
      .value("INTS", AttributeProto::AttributeType::INTS)
      .value("STRINGS", AttributeProto::AttributeType::STRINGS)
      .value("GRAPHS", AttributeProto::AttributeType::GRAPHS)
      .value("SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS)
      .export_values();
  attribute_type
      .def_static(
          "items",
          []() {
            return std::vector<std::pair<std::string, AttributeProto::AttributeType>>{
                {"UNDEFINED", AttributeProto::AttributeType::UNDEFINED},
                {"FLOAT", AttributeProto::AttributeType::FLOAT},
                {"INT", AttributeProto::AttributeType::INT},
                {"STRING", AttributeProto::AttributeType::STRING},
                {"GRAPH", AttributeProto::AttributeType::GRAPH},
                {"SPARSE_TENSOR", AttributeProto::AttributeType::SPARSE_TENSOR},
                {"FLOATS", AttributeProto::AttributeType::FLOATS},
                {"INTS", AttributeProto::AttributeType::INTS},
                {"STRINGS", AttributeProto::AttributeType::STRINGS},
                {"GRAPHS", AttributeProto::AttributeType::GRAPHS},
                {"SPARSE_TENSORS", AttributeProto::AttributeType::SPARSE_TENSORS},
            };
          },
          "Returns the list of (name, type).")
      .def_static(
          "keys",
          []() {
            return std::vector<std::string>{
                "UNDEFINED", "FLOAT", "INT",     "STRING", "GRAPH",          "SPARSE_TENSOR",
                "FLOATS",    "INTS",  "STRINGS", "GRAPHS", "SPARSE_TENSORS",
            };
          },
          "Returns the list of names.")
      .def_static(
          "values",
          []() {
            return std::vector<AttributeProto::AttributeType>{
                AttributeProto::AttributeType::UNDEFINED,
                AttributeProto::AttributeType::FLOAT,
                AttributeProto::AttributeType::INT,
                AttributeProto::AttributeType::STRING,
                AttributeProto::AttributeType::GRAPH,
                AttributeProto::AttributeType::SPARSE_TENSOR,
                AttributeProto::AttributeType::FLOATS,
                AttributeProto::AttributeType::INTS,
                AttributeProto::AttributeType::STRINGS,
                AttributeProto::AttributeType::GRAPHS,
                AttributeProto::AttributeType::SPARSE_TENSORS,
            };
          },
          "Returns the list of types.");

  py_AttributeProto.SHORTEN_CODE(AttributeProto::AttributeType, UNDEFINED)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOAT)
      .SHORTEN_CODE(AttributeProto::AttributeType, INT)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRING)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPH)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSOR)
      .SHORTEN_CODE(AttributeProto::AttributeType, FLOATS)
      .SHORTEN_CODE(AttributeProto::AttributeType, INTS)
      .SHORTEN_CODE(AttributeProto::AttributeType, STRINGS)
      .SHORTEN_CODE(AttributeProto::AttributeType, TENSORS)
      .SHORTEN_CODE(AttributeProto::AttributeType, GRAPHS)
      .SHORTEN_CODE(AttributeProto::AttributeType, SPARSE_TENSORS)
      .PYFIELD_STR(AttributeProto, name)
      .PYFIELD_STR(AttributeProto, ref_attr_name)
      .PYFIELD_STR(AttributeProto, doc_string)
      .def_property(
          "type",
          [](const AttributeProto &self) -> AttributeProto::AttributeType { return self.type_; },
          [](AttributeProto &self, py::object obj) {
            if (py::isinstance<py::int_>(obj)) {
              self.type_ = static_cast<AttributeProto::AttributeType>(obj.cast<int>());
            } else {
              self.type_ = obj.cast<AttributeProto::AttributeType>();
            }
          },
          AttributeProto::DOC_type)
      .PYFIELD_OPTIONAL_FLOAT(AttributeProto, f)
      .PYFIELD_OPTIONAL_INT(AttributeProto, i)
      .PYFIELD_STR_AS_BYTES(AttributeProto, s)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, t)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, sparse_tensor)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, g)
      .PYFIELD_OPTIONAL_PROTO(AttributeProto, tp)
      .PYFIELD(AttributeProto, floats)
      .PYFIELD(AttributeProto, ints)
      .PYFIELD(AttributeProto, strings)
      .PYFIELD(AttributeProto, tensors)
      .PYFIELD(AttributeProto, sparse_tensors)
      .PYFIELD(AttributeProto, graphs);
  PYADD_PROTO_SERIALIZATION(AttributeProto);
  DECLARE_REPEATED_FIELD_PROTO(AttributeProto, rep_ap);
  define_repeated_field_type_proto(rep_ap, rep_ap_proto);

  PYDEFINE_PROTO(m, NodeProto)
      .PYFIELD(NodeProto, input)
      .PYFIELD(NodeProto, output)
      .PYFIELD_STR(NodeProto, name)
      .PYFIELD_STR(NodeProto, op_type)
      .PYFIELD_STR(NodeProto, domain)
      .PYFIELD_STR(NodeProto, overload)
      .PYFIELD(NodeProto, attribute)
      .PYFIELD_STR(NodeProto, doc_string)
      .PYFIELD(NodeProto, metadata_props)
      .PYFIELD(NodeProto, device_configurations);
  PYADD_PROTO_SERIALIZATION(NodeProto);
  DECLARE_REPEATED_FIELD_PROTO(NodeProto, rep_node);
  define_repeated_field_type_proto(rep_node, rep_node_proto);

  PYDEFINE_PROTO(m, GraphProto)
      .PYFIELD(GraphProto, node)
      .PYFIELD_STR(GraphProto, name)
      .PYFIELD(GraphProto, initializer)
      .PYFIELD(GraphProto, sparse_initializer)
      .PYFIELD_STR(GraphProto, doc_string)
      .PYFIELD(GraphProto, input)
      .PYFIELD(GraphProto, output)
      .PYFIELD(GraphProto, value_info)
      .PYFIELD(GraphProto, quantization_annotation)
      .PYFIELD(GraphProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(GraphProto);
  DECLARE_REPEATED_FIELD_PROTO(GraphProto, rep_graph);
  define_repeated_field_type_proto(rep_graph, rep_graph_proto);

  PYDEFINE_PROTO(m, FunctionProto)
      .PYFIELD_STR(FunctionProto, name)
      .PYFIELD(FunctionProto, input)
      .PYFIELD(FunctionProto, output)
      .PYFIELD(FunctionProto, attribute)
      .PYFIELD(FunctionProto, attribute_proto)
      .PYFIELD(FunctionProto, node)
      .PYFIELD_STR(FunctionProto, doc_string)
      .PYFIELD(FunctionProto, opset_import)
      .PYFIELD(FunctionProto, value_info)
      .PYFIELD(FunctionProto, metadata_props);
  PYADD_PROTO_SERIALIZATION(FunctionProto);
  DECLARE_REPEATED_FIELD_PROTO(FunctionProto, rep_function);
  define_repeated_field_type_proto(rep_function, rep_function_proto);

  PYDEFINE_PROTO(m, ModelProto)
      .PYFIELD_STR(ModelProto, producer_name)
      .PYFIELD_STR(ModelProto, producer_version)
      .PYFIELD_STR(ModelProto, domain)
      .PYFIELD(ModelProto, model_version)
      .PYFIELD_STR(ModelProto, doc_string)
      .PYFIELD_OPTIONAL_PROTO(ModelProto, graph)
      .PYFIELD(ModelProto, opset_import)
      .PYFIELD_OPTIONAL_INT(ModelProto, ir_version)
      .PYFIELD(ModelProto, metadata_props)
      .PYFIELD(ModelProto, functions)
      .PYFIELD(ModelProto, configuration);
  PYADD_PROTO_SERIALIZATION(ModelProto);
}
