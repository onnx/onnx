#include <string>
#include <unordered_map>

namespace onnx {

class RuntimeTypeRegistry {
public:
  RuntimeTypeRegistry() { }

  class RegisteredType {
  public:
    RegisteredType(size_t id, std::string name) : id_(id), name_(name) { }
    RegisteredType(const RegisteredType&) = default;
    RegisteredType() = default;
    size_t id() const { return id_; }
  private:
    size_t id_;
    std::string name_;
  };

  using type = RegisteredType;

  void set(size_t id, std::string name) {
    registry_[id] = RegisteredType(id, name);
  }

  RegisteredType get(size_t id) {
    return registry_[id];
  }

private:
  std::unordered_map<size_t, RegisteredType> registry_;
};

}
