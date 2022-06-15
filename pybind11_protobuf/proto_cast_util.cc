#include "pybind11_protobuf/proto_cast_util.h"

#include <Python.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "python/google/protobuf/proto_api.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"

namespace py = pybind11;

using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorDatabase;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DescriptorProto;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FileDescriptor;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::MessageFactory;
using ::google::protobuf::python::PyProto_API;
using ::google::protobuf::python::PyProtoAPICapsuleName;

namespace pybind11_protobuf {
namespace {

std::string PythonPackageForDescriptor(const FileDescriptor* file) {
  std::vector<std::pair<const absl::string_view, std::string>> replacements;
  replacements.emplace_back("/", ".");
  replacements.emplace_back(".proto", "_pb2");
  std::string name = file->name();
  return absl::StrReplaceAll(name, replacements);
}

// Resolves the class name of a descriptor via d->containing_type()
py::object ResolveDescriptor(py::object p, const Descriptor* d) {
  return d->containing_type() ? ResolveDescriptor(p, d->containing_type())
                                    .attr(d->name().c_str())
                              : p.attr(d->name().c_str());
}

// Returns true if an exception is an import error.
bool IsImportError(py::error_already_set& e) {
#if ((PY_MAJOR_VERSION > 3) || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 6))
  return e.matches(PyExc_ImportError) || e.matches(PyExc_ModuleNotFoundError);
#else
  return e.matches(PyExc_ImportError);
#endif
}

// Resolves a sequence of python attrs starting from obj.
// If any does not exist, returns nullopt.
absl::optional<py::object> ResolveAttrs(
    py::handle obj, std::initializer_list<const char*> names) {
  py::object tmp;
  for (const char* name : names) {
    PyObject* attr = PyObject_GetAttrString(obj.ptr(), name);
    if (attr == nullptr) {
      PyErr_Clear();
      return absl::nullopt;
    }
    tmp = py::reinterpret_steal<py::object>(attr);
    obj = py::handle(attr);
  }
  return tmp;
}

// Resolves a single attribute using the python MRO (method resolution order).
// Mimics PyObject_GetAttrString.
//
// Unfortunately the metaclass mechanism used by protos (fast_cpp_protos) does
// not leave __dict__ in a state where the default getattr functions find the
// base class methods, so we resolve those using MRO.
absl::optional<py::object> ResolveAttrMRO(py::handle obj, const char* name) {
  PyObject* attr;
  const auto* t = Py_TYPE(obj.ptr());
  if (!t->tp_mro) {
    PyObject* attr = PyObject_GetAttrString(obj.ptr(), name);
    if (attr) {
      return py::reinterpret_steal<py::object>(attr);
    }
    PyErr_Clear();
    return absl::nullopt;
  }

  auto unicode = py::reinterpret_steal<py::object>(PyUnicode_FromString(name));
  auto bases = py::reinterpret_borrow<py::tuple>(t->tp_mro);
  for (py::handle h : bases) {
    auto base = reinterpret_cast<PyTypeObject*>(h.ptr());
    if (base->tp_getattr) {
      attr = (*base->tp_getattr)(obj.ptr(), const_cast<char*>(name));
      if (attr) {
        return py::reinterpret_steal<py::object>(attr);
      }
      PyErr_Clear();
    }
    if (base->tp_getattro) {
      attr = (*base->tp_getattro)(obj.ptr(), unicode.ptr());
      if (attr) {
        return py::reinterpret_steal<py::object>(attr);
      }
      PyErr_Clear();
    }
  }
  return absl::nullopt;
}

absl::optional<std::string> CastToOptionalString(py::handle src) {
  // Avoid pybind11::cast because it throws an exeption.
  pybind11::detail::make_caster<std::string> c;
  if (c.load(src, false)) {
    return pybind11::detail::cast_op<std::string>(std::move(c));
  }
  return absl::nullopt;
}

#if defined(GOOGLE_PROTOBUF_VERSION)
// The current version, represented as a single integer to make comparison
// easier:  major * 10^6 + minor * 10^3 + micro
uint64_t VersionStringToNumericVersion(absl::string_view version_str) {
  std::vector<absl::string_view> split = absl::StrSplit(version_str, '.');
  uint64_t major = 0, minor = 0, micro = 0;
  if (split.size() == 3 &&  //
      absl::SimpleAtoi(split[0], &major) &&
      absl::SimpleAtoi(split[1], &minor) &&
      absl::SimpleAtoi(split[2], &micro)) {
    return major * 1000000 + minor * 1000 + micro;
  }
  return 0;
}
#endif

class GlobalState {
 public:
  // Global state singleton intentionally leaks at program termination.
  // If destructed along with other static variables, it causes segfaults
  // due to order of destruction conflict with python threads. See
  // https://github.com/pybind/pybind11/issues/1598
  static GlobalState* instance() {
    static auto instance = new GlobalState();
    return instance;
  }

  py::handle global_pool() { return global_pool_; }
  const PyProto_API* py_proto_api() { return py_proto_api_; }
  bool using_fast_cpp() const { return using_fast_cpp_; }

  // Allocate a python proto message instance using the native python
  // allocations.
  py::object PyMessageInstance(const Descriptor* descriptor);

  // Allocates a fast cpp proto python object, also returning
  // the embedded c++ proto2 message type. The returned message
  // pointer cannot be null.
  std::pair<py::object, Message*> PyFastCppProtoMessageInstance(
      const Descriptor* descriptor);

  // Import (and cache) a python module.
  py::module_ ImportCached(const std::string& module_name);

 private:
  GlobalState();

  const PyProto_API* py_proto_api_ = nullptr;
  bool using_fast_cpp_ = false;
  py::object global_pool_;
  py::object factory_;
  py::object find_message_type_by_name_;
  py::object get_prototype_;

  absl::flat_hash_map<std::string, py::module_> import_cache_;
};

GlobalState::GlobalState() {
  assert(PyGILState_Check());

  // pybind11_protobuf casting needs a dependency on proto internals to work.
  try {
    ImportCached("google.protobuf.descriptor");
    auto descriptor_pool =
        ImportCached("google.protobuf.descriptor_pool");
    auto message_factory =
        ImportCached("google.protobuf.message_factory");
    global_pool_ = descriptor_pool.attr("Default")();
    factory_ = message_factory.attr("MessageFactory")(global_pool_);
    find_message_type_by_name_ = global_pool_.attr("FindMessageTypeByName");
    get_prototype_ = factory_.attr("GetPrototype");
  } catch (py::error_already_set& e) {
    if (IsImportError(e)) {
      std::cerr << "Add a python dependency on "
                   "\"@com_google_protobuf//:protobuf_python\"" << std::endl;
    }

    // TODO(pybind11-infra): narrow down to expected exception(s).
    e.restore();
    PyErr_Print();

    global_pool_ = {};
    factory_ = {};
    find_message_type_by_name_ = {};
    get_prototype_ = {};
  }

  // determine the proto implementation.
  auto type =
      ImportCached("google.protobuf.internal.api_implementation")
          .attr("Type")();
  using_fast_cpp_ = (CastToOptionalString(type).value_or("") == "cpp");

#if defined(PYBIND11_PROTOBUF_ENABLE_PYPROTO_API)
  // DANGER: The only way to guarantee that the PyProto_API doesn't have
  // incompatible ABI changes is to ensure that the python protobuf .so
  // and all other extension .so files are built with the exact same
  // environment, including compiler, flags, etc. It's also expected
  // that the global_pool() objects are the same. And there's no way for
  // bazel to do that right now.
  //
  // Otherwise, we're left with (1), the  PyProto_API module reaching into the
  // internals of a potentially incompatible Descriptor type from this CU, (2)
  // this CU reaching into the potentially incompatible internals of PyProto_API
  // implementation, or (3) disabling access to PyProto_API unless compile
  // options suggest otherwise.
  //
  // By default (3) is used, however if the define is set *and* the version
  // matches, then pybind11_protobuf will assume that this will work.
  py_proto_api_ =
      static_cast<PyProto_API*>(PyCapsule_Import(PyProtoAPICapsuleName(), 0));
  if (py_proto_api_ == nullptr) {
    // The module implementing fast cpp protos is not loaded, clear the error.
    assert(!using_fast_cpp_);
    PyErr_Clear();
  }
#else
  py_proto_api_ = nullptr;
  using_fast_cpp_ = false;
#endif

#if defined(GOOGLE_PROTOBUF_VERSION)
  /// The C++ version of PyProto_API must match that loaded by python,
  /// otherwise the details of the underlying implementation may cause
  /// crashes. This limits the ability to pass some protos from C++ to
  /// python.
  if (py_proto_api_) {
    auto version =
        ResolveAttrs(ImportCached("google.protobuf"), {"__version__"});
    std::string version_str =
        version ? CastToOptionalString(*version).value_or("") : "";
    if (GOOGLE_PROTOBUF_VERSION != VersionStringToNumericVersion(version_str)) {
      std::cerr << "Python version " << version_str
                << " does not match C++ version " << GOOGLE_PROTOBUF_VERSION
                << std::endl;
      using_fast_cpp_ = false;
      py_proto_api_ = nullptr;
    }
  }
#endif
}

py::module_ GlobalState::ImportCached(const std::string& module_name) {
  auto cached = import_cache_.find(module_name);
  if (cached != import_cache_.end()) {
    return cached->second;
  }
  auto module = py::module_::import(module_name.c_str());
  import_cache_[module_name] = module;
  return module;
}

py::object GlobalState::PyMessageInstance(const Descriptor* descriptor) {
  auto module_name = PythonPackageForDescriptor(descriptor->file());
  if (!module_name.empty()) {
    auto cached = import_cache_.find(module_name);
    if (cached != import_cache_.end()) {
      return ResolveDescriptor(cached->second, descriptor)();
    }
  }

  // First attempt to construct the proto from the global pool.
  if (global_pool_) {
    try {
      auto d = find_message_type_by_name_(descriptor->full_name());
      auto p = get_prototype_(d);
      return p();
    } catch (...) {
      // TODO(pybind11-infra): narrow down to expected exception(s).
      PyErr_Clear();
    }
  }

  // If that fails, attempt to import the module.
  if (!module_name.empty()) {
    try {
      return ResolveDescriptor(ImportCached(module_name), descriptor)();
    } catch (py::error_already_set& e) {
      // TODO(pybind11-infra): narrow down to expected exception(s).
      e.restore();
      PyErr_Print();
    }
  }

  throw py::type_error("Cannot construct a protocol buffer message type " +
                       descriptor->full_name() +
                       " in python. Is there a missing dependency on module " +
                       module_name + "?");
}

std::pair<py::object, Message*> GlobalState::PyFastCppProtoMessageInstance(
    const Descriptor* descriptor) {
  assert(descriptor != nullptr);
  assert(py_proto_api_ != nullptr);

  // Create a PyDescriptorPool, temporarily, it will be used by the NewMessage
  // API call which will store it in the classes it creates.
  //
  // Note: Creating Python classes is a bit expensive, it might be a good idea
  // for client code to create the pool once, and store it somewhere along with
  // the C++ pool; then Python pools and classes are cached and reused.
  // Otherwise, consecutives calls to this function may or may not reuse
  // previous classes, depending on whether the returned instance has been
  // kept alive.
  //
  // IMPORTANT CAVEAT: The C++ DescriptorPool must not be deallocated while
  // there are any messages using it.
  // Furthermore, since the cache uses the DescriptorPool address, allocating
  // a new DescriptorPool with the same address is likely to use dangling
  // pointers.
  // It is probably better for client code to keep the C++ DescriptorPool alive
  // until the end of the process.
  // TODO(amauryfa): Add weakref or on-deletion callbacks to C++ DescriptorPool.
  py::object descriptor_pool = py::reinterpret_steal<py::object>(
      py_proto_api_->DescriptorPool_FromPool(descriptor->file()->pool()));
  if (descriptor_pool.ptr() == nullptr) {
    throw py::error_already_set();
  }

  py::object result = py::reinterpret_steal<py::object>(
      py_proto_api_->NewMessage(descriptor, nullptr));
  if (result.ptr() == nullptr) {
    throw py::error_already_set();
  }
  Message* message = py_proto_api_->GetMutableMessagePointer(result.ptr());
  if (message == nullptr) {
    throw py::error_already_set();
  }
  return {std::move(result), message};
}

// Create C++ DescriptorPools based on Python DescriptorPools.
// The Python pool will provide message definitions when they are needed.
// This gives an efficient way to create C++ Messages from Python definitions.
class PythonDescriptorPoolWrapper {
 public:
  // The singleton which handles multiple wrapped pools.
  // It is never deallocated, but data corresponding to a Python pool
  // is cleared when the pool is destroyed.
  static PythonDescriptorPoolWrapper* instance() {
    static auto instance = new PythonDescriptorPoolWrapper();
    return instance;
  }

  // To build messages these 3 objects often come together:
  // - a DescriptorDatabase provides the representation of .proto files.
  // - a DescriptorPool manages the live descriptors with cross-linked pointers.
  // - a MessageFactory manages the proto instances and their memory layout.
  struct Data {
    std::unique_ptr<DescriptorDatabase> database;
    std::unique_ptr<const DescriptorPool> pool;
    std::unique_ptr<MessageFactory> factory;
  };

  // Return (and maybe create) a C++ DescriptorPool that corresponds to the
  // given Python DescriptorPool.
  // The returned pointer has the same lifetime as the Python DescriptorPool:
  // its data will be deleted when the Python object is deleted.
  const Data* GetPoolFromPythonPool(py::handle python_pool) {
    PyObject* key = python_pool.ptr();
    // Get or create an entry for this key.
    auto& pool_entry = pools_map[key];
    if (pool_entry.database) {
      // Found in cache, return it.
      return &pool_entry;
    }

    // An attempt at cleanup could be made by using a py::weakref to the
    // underlying python pool, and removing the map entry when the pool
    // disappears, that is fundamentally unsafe because (1) a cloned c++ object
    // may outlive the python pool, and (2) for the fast_cpp_proto case, there's
    // no support for weak references.

    auto database = absl::make_unique<DescriptorPoolDatabase>(
        py::reinterpret_borrow<py::object>(python_pool));
    auto pool = absl::make_unique<DescriptorPool>(database.get());
    auto factory = absl::make_unique<DynamicMessageFactory>(pool.get());
    // When wrapping the Python descriptor_poool.Default(), apply an important
    // optimization:
    // - the pool is based on the C++ generated_pool(), so that compiled
    //   C++ modules can be found without using the DescriptorDatabase and
    //   the Python DescriptorPool.
    // - the MessageFactory returns instances of C++ compiled messages when
    //   possible: some methods are much more optimized, and the created
    //   Message can be cast to the C++ class.  We use this last property in
    //   the proto_caster class.
    // This is done only for the Default pool, because generated C++ modules
    // and generated Python modules are built from the same .proto sources.
    if (python_pool.is(GlobalState::instance()->global_pool())) {
      pool->internal_set_underlay(DescriptorPool::generated_pool());
      factory->SetDelegateToGeneratedFactory(true);
    }

    // Cache the created objects.
    pool_entry = Data{std::move(database), std::move(pool), std::move(factory)};
    return &pool_entry;
  }

 private:
  PythonDescriptorPoolWrapper() = default;

  // Similar to DescriptorPoolDatabase: wraps a Python DescriptorPool
  // as a DescriptorDatabase.
  class DescriptorPoolDatabase : public DescriptorDatabase {
   public:
    DescriptorPoolDatabase(py::object python_pool)
        : pool_(std::move(python_pool)) {}

    // These 3 methods implement DescriptorDatabase and delegate to
    // the Python DescriptorPool.

    // Find a file by file name.
    bool FindFileByName(const std::string& filename,
                        FileDescriptorProto* output) override {
      try {
        auto file = pool_.attr("FindFileByName")(filename);
        return CopyToFileDescriptorProto(file, output);
      } catch (py::error_already_set& e) {
        std::cerr << "FindFileByName " << filename << " raised an error";

        // This prints and clears the error.
        e.restore();
        PyErr_Print();
      }
      return false;
    }

    // Find the file that declares the given fully-qualified symbol name.
    bool FindFileContainingSymbol(const std::string& symbol_name,
                                  FileDescriptorProto* output) override {
      try {
        auto file = pool_.attr("FindFileContainingSymbol")(symbol_name);
        return CopyToFileDescriptorProto(file, output);
      } catch (py::error_already_set& e) {
        std::cerr << "FindFileContainingSymbol " << symbol_name
                   << " raised an error";

        // This prints and clears the error.
        e.restore();
        PyErr_Print();
      }
      return false;
    }

    // Find the file which defines an extension extending the given message type
    // with the given field number.
    bool FindFileContainingExtension(const std::string& containing_type,
                                     int field_number,
                                     FileDescriptorProto* output) override {
      try {
        auto descriptor = pool_.attr("FindMessageTypeByName")(containing_type);
        auto file =
            pool_.attr("FindExtensionByNymber")(descriptor, field_number)
                .attr("file");
        return CopyToFileDescriptorProto(file, output);
      } catch (py::error_already_set& e) {
        std::cerr << "FindFileContainingExtension " << containing_type << " "
                   << field_number << " raised an error";

        // This prints and clears the error.
        e.restore();
        PyErr_Print();
      }
      return false;
    }

   private:
    bool CopyToFileDescriptorProto(py::handle py_file_descriptor,
                                   FileDescriptorProto* output) {
      if (GlobalState::instance()->py_proto_api()) {
        try {
          py::object c_proto = py::reinterpret_steal<py::object>(
              GlobalState::instance()
                  ->py_proto_api()
                  ->NewMessageOwnedExternally(output, nullptr));
          if (c_proto) {
            py_file_descriptor.attr("CopyToProto")(c_proto);
            return true;
          }
        } catch (py::error_already_set& e) {
          std::cerr << "CopyToFileDescriptorProto raised an error";

          // This prints and clears the error.
          e.restore();
          PyErr_Print();
        }
      }

      py::object wire = py_file_descriptor.attr("serialized_pb");
      const char* bytes = PYBIND11_BYTES_AS_STRING(wire.ptr());
      return output->ParsePartialFromArray(bytes,
                                           PYBIND11_BYTES_SIZE(wire.ptr()));
    }

    py::object pool_;  // never dereferenced.
  };

  // This map caches the wrapped objects, indexed by DescriptorPool address.
  absl::flat_hash_map<PyObject*, Data> pools_map;
};

}  // namespace

void InitializePybindProtoCastUtil() {
  assert(PyGILState_Check());
  GlobalState::instance();
}

void ImportProtoDescriptorModule(const Descriptor* descriptor) {
  assert(PyGILState_Check());
  if (!descriptor) return;
  auto module_name = PythonPackageForDescriptor(descriptor->file());
  if (module_name.empty()) return;
  try {
    GlobalState::instance()->ImportCached(module_name);
  } catch (py::error_already_set& e) {
    if (IsImportError(e)) {
      std::cerr << "Python module " << module_name << " unavailable."
                 << std::endl;
    } else {
      std::cerr << "ImportDescriptorModule raised an error";
      // This prints and clears the error.
      e.restore();
      PyErr_Print();
    }
  }
}

const Message* PyProtoGetCppMessagePointer(py::handle src) {
  assert(PyGILState_Check());
  if (!GlobalState::instance()->py_proto_api()) return nullptr;
  auto* ptr =
      GlobalState::instance()->py_proto_api()->GetMessagePointer(src.ptr());
  if (ptr == nullptr) {
    // Clear the type_error set by GetMessagePointer sets a type_error when
    // src was not a wrapped C++ proto message.
    PyErr_Clear();
    return nullptr;
  }
  return ptr;
}

absl::optional<std::string> PyProtoDescriptorName(py::handle py_proto) {
  assert(PyGILState_Check());
  auto py_full_name = ResolveAttrs(py_proto, {"DESCRIPTOR", "full_name"});
  if (py_full_name) {
    return CastToOptionalString(*py_full_name);
  }
  return absl::nullopt;
}

bool PyProtoIsCompatible(py::handle py_proto, const Descriptor* descriptor) {
  assert(PyGILState_Check());
  assert(descriptor->file()->pool() == DescriptorPool::generated_pool());

  auto py_descriptor = ResolveAttrs(py_proto, {"DESCRIPTOR"});
  if (!py_descriptor) {
    // Not a valid protobuf -- missing DESCRIPTOR.
    return false;
  }

  // Test full_name equivalence.
  {
    auto py_full_name = ResolveAttrs(*py_descriptor, {"full_name"});
    if (!py_full_name) {
      // Not a valid protobuf -- missing DESCRIPTOR.full_name
      return false;
    }
    auto full_name = CastToOptionalString(*py_full_name);
    if (!full_name || *full_name != descriptor->full_name()) {
      // Name mismatch.
      return false;
    }
  }

  // The C++ descriptor is compiled in (see above assert), so the py_proto
  // is expected to be from the global pool, i.e. the DESCRIPTOR.file.pool
  // instance is the global python pool, and not a custom pool.
  auto py_pool = ResolveAttrs(*py_descriptor, {"file", "pool"});
  if (py_pool) {
    return py_pool->is(GlobalState::instance()->global_pool());
  }

  // The py_proto is missing a DESCRIPTOR.file.pool, but the name matches.
  // This will not happen with a native python implementation, but does
  // occur with the deprecated :proto_casters, and could happen with other
  // mocks.  Returning true allows the caster to call PyProtoCopyToCProto.
  return true;
}

bool PyProtoCopyToCProto(py::handle py_proto, Message* message) {
  assert(PyGILState_Check());
  auto serialize_fn = ResolveAttrMRO(py_proto, "SerializePartialToString");
  if (!serialize_fn) {
    throw py::type_error(
        "SerializePartialToString method not found; is this a " +
        message->GetDescriptor()->full_name());
  }
  auto wire = (*serialize_fn)();
  const char* bytes = PYBIND11_BYTES_AS_STRING(wire.ptr());
  if (!bytes) {
    throw py::type_error("SerializePartialToString failed; is this a " +
                         message->GetDescriptor()->full_name());
  }
  return message->ParsePartialFromArray(bytes, PYBIND11_BYTES_SIZE(wire.ptr()));
}

void CProtoCopyToPyProto(Message* message, py::handle py_proto) {
  assert(PyGILState_Check());
  auto merge_fn = ResolveAttrMRO(py_proto, "MergeFromString");
  if (!merge_fn) {
    throw py::type_error("MergeFromString method not found; is this a " +
                         message->GetDescriptor()->full_name());
  }

  auto serialized = message->SerializePartialAsString();
#if PY_MAJOR_VERSION >= 3
  auto view = py::memoryview::from_memory(serialized.data(), serialized.size());
#else
  py::bytearray view(serialized);
#endif
  (*merge_fn)(view);
}

std::unique_ptr<Message> AllocateCProtoFromPythonSymbolDatabase(
    py::handle src, const std::string& full_name) {
  assert(PyGILState_Check());
  auto pool = ResolveAttrs(src, {"DESCRIPTOR", "file", "pool"});
  if (!pool) {
    throw py::type_error("Object is not a valid protobuf");
  }

  auto pool_data =
      PythonDescriptorPoolWrapper::instance()->GetPoolFromPythonPool(*pool);
  // The following call will query the DescriptorDatabase, which fetches the
  // necessary Python descriptors and feeds them into the C++ pool.
  // The result stays cached as long as the Python pool stays alive.
  const Descriptor* descriptor =
      pool_data->pool->FindMessageTypeByName(full_name);
  if (!descriptor) {
    throw py::type_error("Could not find descriptor: " + full_name);
  }
  const Message* prototype = pool_data->factory->GetPrototype(descriptor);
  if (!prototype) {
    throw py::type_error("Unable to get prototype for " + full_name);
  }
  return std::unique_ptr<Message>(prototype->New());
}

namespace {

std::string ReturnValuePolicyName(py::return_value_policy policy) {
  switch (policy) {
    case py::return_value_policy::automatic:
      return "automatic";
    case py::return_value_policy::automatic_reference:
      return "automatic_reference";
    case py::return_value_policy::take_ownership:
      return "take_ownership";
    case py::return_value_policy::copy:
      return "copy";
    case py::return_value_policy::move:
      return "move";
    case py::return_value_policy::reference:
      return "reference";
    case py::return_value_policy::reference_internal:
      return "reference_internal";
    default:
      return "INVALID_ENUM_VALUE";
  }
}

}  // namespace

py::handle GenericPyProtoCast(Message* src, py::return_value_policy policy,
                              py::handle parent, bool is_const) {
  assert(src != nullptr);
  assert(PyGILState_Check());
  auto py_proto =
      GlobalState::instance()->PyMessageInstance(src->GetDescriptor());

  CProtoCopyToPyProto(src, py_proto);
  return py_proto.release();
}

py::handle GenericFastCppProtoCast(Message* src, py::return_value_policy policy,
                                   py::handle parent, bool is_const) {
  assert(policy != pybind11::return_value_policy::automatic);
  assert(policy != pybind11::return_value_policy::automatic_reference);
  assert(src != nullptr);
  assert(PyGILState_Check());
  assert(GlobalState::instance()->py_proto_api() != nullptr);

  switch (policy) {
    case py::return_value_policy::move:
    case py::return_value_policy::take_ownership: {
      std::pair<py::object, Message*> descriptor_pair =
          GlobalState::instance()->PyFastCppProtoMessageInstance(
              src->GetDescriptor());
      py::object& result = descriptor_pair.first;
      Message* result_message = descriptor_pair.second;

      if (result_message->GetReflection() == src->GetReflection()) {
        // The internals may be Swapped iff the protos use the same Reflection
        // instance.
        result_message->GetReflection()->Swap(src, result_message);
      } else {
        auto serialized = src->SerializePartialAsString();
        if (!result_message->ParseFromString(serialized)) {
          throw py::type_error(
              "Failed to copy protocol buffer with mismatched descriptor");
        }
      }
      return result.release();
    } break;

    case py::return_value_policy::copy: {
      std::pair<py::object, Message*> descriptor_pair =
          GlobalState::instance()->PyFastCppProtoMessageInstance(
              src->GetDescriptor());
      py::object& result = descriptor_pair.first;
      Message* result_message = descriptor_pair.second;

      if (result_message->GetReflection() == src->GetReflection()) {
        // The internals may be copied iff the protos use the same Reflection
        // instance.
        result_message->CopyFrom(*src);
      } else {
        auto serialized = src->SerializePartialAsString();
        if (!result_message->ParseFromString(serialized)) {
          throw py::type_error(
              "Failed to copy protocol buffer with mismatched descriptor");
        }
      }
      return result.release();
    } break;

    case py::return_value_policy::reference:
    case py::return_value_policy::reference_internal: {
      // NOTE: Reference to const are currently unsafe to return.
      py::object result = py::reinterpret_steal<py::object>(
          GlobalState::instance()->py_proto_api()->NewMessageOwnedExternally(
              src, nullptr));
      if (policy == py::return_value_policy::reference_internal) {
        py::detail::keep_alive_impl(result, parent);
      }
      return result.release();
    } break;

    default:
      std::string message("pybind11_protobuf unhandled return_value_policy::");
      throw py::cast_error(message + ReturnValuePolicyName(policy));
  }
}

py::handle GenericProtoCast(Message* src, py::return_value_policy policy,
                            py::handle parent, bool is_const) {
  assert(src != nullptr);
  assert(PyGILState_Check());

  // Return a native python-allocated proto when:
  // 1. The binary does not have a py_proto_api instance, or
  // 2. a) the proto is from the default pool and
  //    b) the binary is not using fast_cpp_protos.
  if ((GlobalState::instance()->py_proto_api() == nullptr) ||
      (src->GetDescriptor()->file()->pool() ==
           DescriptorPool::generated_pool() &&
       !GlobalState::instance()->using_fast_cpp())) {
    return GenericPyProtoCast(src, policy, parent, is_const);
  }

  // If this is a dynamically generated proto, then we're going to need to
  // construct a mapping between C++ pool() and python pool(), and then
  // use the PyProto_API to make it work.
  return GenericFastCppProtoCast(src, policy, parent, is_const);
}

}  // namespace pybind11_protobuf
