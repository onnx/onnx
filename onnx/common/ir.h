// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/string_utils.h"
#include "onnx/common/array_ref.h"
#include "onnx/common/assertions.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/graph_node_list.h"
#include "onnx/common/tensor.h"


#define ONNX_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete


namespace ONNX_NAMESPACE {

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside it.
// All references inside the graph are raw pointers.
// Destroying the Graph will invalidate any pointers to nodes in the graph.
struct Graph;


// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;


// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;


class ResourceGuard final {
  std::function<void()> destructor_;
  bool released_;

public:
  ResourceGuard(std::function<void()> destructor)
    : destructor_(std::move(destructor))
    , released_(false) {}

  ~ResourceGuard() {
    if (!released_) destructor_();
  }

  void release() {
    released_ = true;
  }
};


struct Dimension final {
  Dimension(int dim)
    : is_int(true), dim(dim) {
  }
  Dimension(std::string param)
    : is_int(false), dim(-1), param(std::move(param)) {
  }

  bool is_int;
  int64_t dim;
  std::string param;
};


enum class AttributeKind : uint8_t {
  // float, float list, int, int list, string, string list,
  // tensor, tensor list, subgraph, subgraph list
  f, fs, i, is, s, ss, t, ts, g, gs
};


static inline const char * toString(AttributeKind kind) {
  static constexpr const char* names[] = {"f","fs", "i", "is", "s", "ss", "t", "ts", "g", "gs"};
  ONNX_ASSERT(size_t(kind) < sizeof(names) / sizeof(AttributeKind));
  return names[int(kind)];
}


struct AttributeValue {
  AttributeValue(Symbol name)
  : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() = default;
};


template<typename T, AttributeKind Kind>
struct ScalarAttributeValue final : public AttributeValue {
  using ConstructorType = const T &;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ConstructorType value_)
  : AttributeValue(name), value_(value_) {}
  ValueType & value() {
    return value_;
  }
  virtual Ptr clone() const override {
    return Ptr(new ScalarAttributeValue(name, value_));
  }
  virtual AttributeKind kind() const override { return Kind; }

private:
  ValueType value_;
};


template<typename T, AttributeKind Kind>
struct VectorAttributeValue final : public AttributeValue {
  using ConstructorType = const std::vector<T> &&;
  using ValueType = std::vector<T>;
  VectorAttributeValue(Symbol name, ConstructorType value_)
  : AttributeValue(name), value_(std::move(value_)) {}
  ValueType & value() {
    return value_;
  }
  virtual AttributeKind kind() const override { return Kind; }
  virtual std::unique_ptr<AttributeValue> clone() const override {
    auto copy = value_;
    return Ptr(new VectorAttributeValue(name, std::move(copy)));
  }
private:
  ValueType value_;
};


using FloatAttr = ScalarAttributeValue<double,AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double,AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t,AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t,AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string,AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string,AttributeKind::ss>;
using TensorAttr = ScalarAttributeValue<Tensor,AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<Tensor,AttributeKind::ts>;
using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>,AttributeKind::g>;
using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>,AttributeKind::gs>;


// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node * n = g->create(kSelect)->set_i(kOffset,3)->set_f(kValue,3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template<typename Derived>
struct Attributes {
  Attributes() {}
  void copyAttributes(const Attributes & rhs) {
    values_.clear();
    values_.reserve(rhs.values_.size());
    for(auto & i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  bool hasAttribute(Symbol name) const {
    return find(name,false) != values_.end();
  }
  AttributeKind kindOf(Symbol name) const {
    return (*find(name,true))->kind();
  }
  Derived* removeAttribute(Symbol name) {
    values_.erase(find(name,true));
    return This();
  }
  bool hasAttributes() const {
    return values_.size() > 0;
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    names.reserve(values_.size());
    for(auto & a : values_)
      names.push_back(a->name);
    return names;
  }

  #define CREATE_ACCESSOR(Kind, method) \
  Derived* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    return set<Kind##Attr>(name,std::forward<Kind##Attr::ConstructorType>(v)); \
  } \
  const Kind##Attr::ValueType& method(Symbol name) const { \
    return get<Kind##Attr>(name); \
  }
  CREATE_ACCESSOR(Float,f)
  CREATE_ACCESSOR(Floats,fs)
  CREATE_ACCESSOR(String,s)
  CREATE_ACCESSOR(Strings,ss)
  CREATE_ACCESSOR(Int,i)
  CREATE_ACCESSOR(Ints,is)
  CREATE_ACCESSOR(Tensor,t)
  CREATE_ACCESSOR(Tensors,ts)
  CREATE_ACCESSOR(Graph,g)
  CREATE_ACCESSOR(Graphs,gs)

  #undef CREATE_ACCESSOR

private:
  Derived* This() {
    return static_cast<Derived*>(this);
  }
  template<typename T>
  Derived* set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    if(it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return This();
  }
  template<typename T>
  typename T::ValueType & get(Symbol name) const {
    auto it = find(name, true);
    T* child = static_cast<T*>(it->get());
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name, bool required) {
    auto it = std::find_if(values_.begin(), values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    ONNX_ASSERT(!required || it != values_.end());
    return it;
  }
  using const_iterator = std::vector<AVPtr>::const_iterator;
  const_iterator find(Symbol name, bool required) const {
    auto it = std::find_if(values_.begin(), values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    ONNX_ASSERTM(!required || it != values_.end(),
        "%s:%u: %s: required undefined attribute '%s'", __FILE__, __LINE__, __func__, name.toString());
    return it;
  }
};



// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use final {
  Use(Node * user, size_t offset)
  : user(user), offset(offset) {}
  Node * user;
  size_t offset;
};

static inline bool operator==(const Use & a, const Use & b) {
  return a.user == b.user && a.offset == b.offset;
}


// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
using NodeKind = Symbol;


struct Value final {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node * node_, size_t offset_);

private:
  friend struct Node;
  friend struct Graph;
  Node * node_;
  size_t offset_;
  size_t unique_ = 0;          // unique id
  size_t stage_ = 0;           // 0-forward, 1-backward, 2-double-backward,...
  use_list uses_;
  bool has_unique_name_;
  std::string unique_name_;
  ONNX_NAMESPACE::TensorProto_DataType elem_type_;
  std::vector<Dimension> sizes_;

public:
  Value* setElemType(ONNX_NAMESPACE::TensorProto_DataType elem_type) {
    elem_type_ = elem_type;
    return this;
  }
  ONNX_NAMESPACE::TensorProto_DataType elemType() const {
    return elem_type_;
  }
  Value* setSizes(std::vector<Dimension> sizes) {
    sizes_ = std::move(sizes);
    return this;
  }
  const std::vector<Dimension>& sizes() const {
    return sizes_;
  }
  size_t unique() const {
    return unique_;
  }
  bool has_unique_name() const {
    return has_unique_name_;
  }
  std::string uniqueName() const {
    if(has_unique_name())
      return unique_name_;
    return ONNX_NAMESPACE::to_string(unique());
  }
  Value* setUniqueName(std::string name) {
    has_unique_name_ = true;
    unique_name_ = std::move(name);
    return this;
  }
  Value* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  size_t stage() const {
    return stage_;
  }
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  const Node * node() const {
    return node_;
  }
  Graph * owningGraph();
  const Graph * owningGraph() const;
  // TODO: make this more const correct
  const use_list & uses() const {
    return uses_;
  }

  // Replaces all uses of this node with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  void replaceAllUsesWith(Value * newValue);

  Value* copyMetadata(Value * from) {
    setElemType(from->elemType());
    setSizes(from->sizes());
    if (from->has_unique_name()) {
      setUniqueName(from->uniqueName());
    }
    return this;
  }

};


struct Node : public Attributes<Node> {
  ONNX_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

private:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the sentinel for the beginning and end of the list
  // such that the list never has null pointers
  // next_in_graph[0] is next pointer
  // next_in_graph[1] is prev pointer
  // using an array to allow the same iterator class for forward and reverse node lists
  // This list represents a topological sort

  Node* next_in_graph[2] = { nullptr, nullptr };
  Node* & next() { return next_in_graph[kNextDirection]; }
  Node* & prev() { return next_in_graph[kPrevDirection]; }
  Node* const & next() const { return next_in_graph[kNextDirection]; }
  Node* const & prev() const { return next_in_graph[kPrevDirection]; }

  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  Graph* graph_;
  size_t stage_;
  bool has_name_;
  std::string name_;
  bool has_doc_string_;
  std::string doc_string_;

protected:
  Node(Graph * graph_, NodeKind kind_); //defined after graph

public:
  bool has_name() {
    return has_name_;
  }
  const std::string& name() {
    return name_;
  }
  void setName(std::string name) {
    has_name_ = true;
    name_ = std::move(name);
  }
  bool has_doc_string() const {
    return has_doc_string_;
  }
  const std::string& docString() {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  NodeKind kind() const {
    return kind_;
  }
  Graph * owningGraph() {
    return graph_;
  }
  const Graph * owningGraph() const {
    return graph_;
  }
  size_t stage() const {
    return stage_;
  }
  Node* setStage(size_t s) {
    stage_ = s;
    return this;
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> inputs() {
    return inputs_;
  }
  ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  ArrayRef<Value*> outputs() {
    return outputs_;
  }
  ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  bool hasUses() const {
    for(auto o : outputs()) {
      if(o->uses().size() > 0)
        return true;
    }
    return false;
  }
  void replaceAllUsesWith(Node * n) {
    ONNX_ASSERT(outputs().size() == n->outputs().size());
    size_t nOutputs = outputs().size();
    for(size_t i = 0; i < nOutputs; i++) {
      outputs()[i]->replaceAllUsesWith(n->outputs()[i]);
    }
  }
  // lots of things like chunk have a single input or singel output, so we have a
  // helper to make accessing it easier
  Value * input() {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value * output() {
    ONNX_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value * input() const {
    ONNX_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value * input(size_t i) {
    return inputs_.at(i);
  }
  const Value * input(size_t i) const {
    return inputs_.at(i);
  }

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value * node) {
    ONNX_ASSERT(graph_ == node->owningGraph());
    node->uses_.emplace_back(this, inputs_.size());
    inputs_.push_back(node);
    return node;
  }

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value * replaceInput(size_t i, Value * newValue) {
    ONNX_ASSERT(newValue->owningGraph() == graph_);
    Value * old = dropInput(i);
    inputs_[i] = newValue;
    newValue->uses_.emplace_back(this, i);
    return old;
  }

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value * from, Value * to) {
    ONNX_ASSERT(from->owningGraph() == graph_);
    ONNX_ASSERT(to->owningGraph() == graph_);
    size_t i = 0;
    for(auto input : inputs()) {
      if(input == from)
        replaceInput(i, to);
      i++;
    }
  }

  Value* addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    return outputs_.back();
  }

  void eraseOutput(size_t i);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node * n) {
    ONNX_ASSERT(n->inGraphList());
    insertAfter(n->prev());
    return this;
  }

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node * n) {
    ONNX_ASSERT(!inGraphList() && n->inGraphList());
    Node * next = n->next();
    n->next() = this;
    this->prev() = n;
    this->next() = next;
    next->prev() = this;
    return this;
  }

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node * n) {
    removeFromList();
    insertAfter(n);
  }

  // Move a node 'n' (already in the graph) before 'this' in the topological order.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node * n) {
    removeFromList();
    insertBefore(n);
  }

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i) {
    dropInput(i);
    // everything after this input shifts left,
    // so we need to update their use offsets to match
    for(size_t j = i+1; j < inputs_.size(); j++) {
      auto it = findUseForInput(j);
      it->offset--;
    }
    inputs_.erase(inputs_.begin() + i);
  }

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs() {
    for(size_t i = 0; i < inputs().size(); ++i)
      dropInput(i);
    inputs_.clear();
  }

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  graph_node_list_iterator iterator();
  graph_node_list_iterator reverseIterator();
  const_graph_node_list_iterator iterator() const;
  const_graph_node_list_iterator reverseIterator() const;

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  // TODO: Make this const correct
  template<typename T>
  T* cast() {
    if(T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template<typename T>
  T* expect() {
    ONNX_ASSERTM(T::Kind == kind(), "expected a %s but found a %s", T::Kind.toString(), kind().toString());
    return static_cast<T*>(this);
  }

  virtual ~Node() = default;

private:
  // Lookup iterator in use list of _input i_ that corresponds to its use of _this_
  use_list::iterator findUseForInput(size_t i) {
    auto & input_uses = inputs_[i]->uses_;
    // O(N) on the use list, but unless we get nodes with +100 uses
    // vector traversal still is probably faster than linked list
    auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
    ONNX_ASSERT(use_it != input_uses.end());
    return use_it;
  }

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i) {
    ONNX_ASSERT(i < inputs_.size());
    auto input_node = inputs_[i];
    auto use_it = findUseForInput(i);
    input_node->uses_.erase(use_it);
    inputs_[i] = nullptr;
    return input_node;
  }

  bool inGraphList() const {
    ONNX_ASSERT(next() != nullptr || prev() == nullptr);
    return next() != nullptr;
  }
  void removeFromList() {
    ONNX_ASSERT(inGraphList());
    Node * next = this->next();
    Node * prev = this->prev();
    prev->next() = next;
    next->prev() = prev;
    this->next() = nullptr;
    this->prev() = nullptr;
  }

protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node * allocNewInstance(Graph * g) {
    return new Node(g, kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional information to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  //
  // NB: This does NOT clone stages.  You're expected to set the stage correctly
  // if you are going to preserve it.
  virtual void cloneFrom(Node * s) {
    copyAttributes(*s);
  }
};

struct Graph final {
ONNX_DISALLOW_COPY_AND_ASSIGN(Graph);
friend struct Node;
friend struct Value;

private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  size_t next_unique_;

  size_t new_node_stage_;

  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node * const output_;
  Node * const input_;

  std::vector<Tensor> initializers_;
  std::vector<std::string> initializer_names_;

  bool has_name_;
  std::string name_;
  bool has_doc_string_;
  std::string doc_string_;

public:
  Graph()
  : next_unique_(0)
  , new_node_stage_(0)
  , output_(initOutput(create(kReturn, 0)))
  , input_(create(kParam, 0))
  , has_name_(false)
  , has_doc_string_(false) {}

  bool has_doc_string() {
    return has_doc_string_;
  }
  const std::string& docString() {
    return doc_string_;
  }
  void setDocString(std::string doc_string) {
    has_doc_string_ = true;
    doc_string_ = std::move(doc_string);
  }
  void addInitializer(Tensor initializer, std::string name) {
    initializers_.push_back(std::move(initializer));
    initializer_names_.push_back(std::move(name));
  }
  void clearInitializers() {
    initializers_.clear();
  }
  const std::vector<Tensor>& initializers() {
    return initializers_;
  }
  const std::vector<std::string>& initializer_names() {
    return initializer_names_;
  }
  ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  ArrayRef<const Value*> inputs() const {
    const auto & inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return graph_node_list(output_, kNextDirection);
  }
  const_graph_node_list nodes() const {
    return const_graph_node_list(output_, kNextDirection);
  }
  // These invocations of begin() on output of function are OK
  // because graph_node_list is non-owning, so it doesn't matter
  // if it immediately dies after the invocation.
  graph_node_list_iterator begin() {
    return nodes().begin();
  }
  const_graph_node_list_iterator begin() const {
    return nodes().begin();
  }
  graph_node_list_iterator end() {
    return nodes().end();
  }
  const_graph_node_list_iterator end() const {
    return nodes().end();
  }
  graph_node_list_iterator rbegin() {
    return nodes().rbegin();
  }
  const_graph_node_list_iterator rbegin() const {
    return nodes().rbegin();
  }
  graph_node_list_iterator rend() {
    return nodes().rend();
  }
  const_graph_node_list_iterator rend() const {
    return nodes().rend();
  }
  Node * return_node() {
    return output_;
  }
  const Node * return_node() const {
    return output_;
  }

  Value * addInput() {
    return input_->addOutput();
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  void advanceStage() {
    new_node_stage_++;
  }
  void setStage(size_t new_stage) {
    new_node_stage_ = new_stage;
  }
  size_t stage() const {
    return new_node_stage_;
  }
  ResourceGuard setStageTemporary(size_t s) {
    auto prev_stage = new_node_stage_;
    new_node_stage_ = s;
    return ResourceGuard([prev_stage, this]() { this->new_node_stage_ = prev_stage; });
  }

  size_t registerOutput(Value * n) {
    output_->addInput(n);
    return outputs().size() - 1;
  }

  Node * create(NodeKind kind, size_t num_outputs=1) {
    // NB: Node constructor adds node to all_nodes
    auto n = new Node(this, kind);
    for(size_t i = 0; i < num_outputs; i++)
      n->addOutput();
    return n;
  }

  Node * create(NodeKind kind, ArrayRef<Value*> inputs, size_t num_outputs=1) {
    auto n = create(kind, num_outputs);
    for(auto i : inputs)
      n->addInput(i);
    return n;
  }

  Node * appendNode(Node * n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertBefore(output_);
    return n;
  }

  Node * prependNode(Node * n) {
    ONNX_ASSERT(n->graph_ == this && !n->inGraphList());
    n->insertAfter(output_);
    return n;
  }

  ~Graph() {
    for (const Node * n : all_nodes)
      delete n;
    for (const Value * v : all_values)
      delete v;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << *this;
    return oss.str();
  }

  bool has_name() const {
    return has_name_;
  }

  const std::string& name() const {
    return name_;
  }

  void setName(std::string name) {
    has_name_ = true;
    name_ = name;
  }

  friend std::ostream& operator<<(std::ostream & out, const Graph & g);

private:

  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    p->setStage(std::numeric_limits<size_t>::max());
    return p;
  }

  void freeNode(Node * n) {
    auto it = all_nodes.find(n);
    ONNX_ASSERT(it != all_nodes.end());
    delete *it;
    all_nodes.erase(it);
  }
  void freeValue(Value * v) {
    auto it = all_values.find(v);
    ONNX_ASSERT(it != all_values.end());
    all_values.erase(it);
  }
};

inline Value::Value(Node * node_, size_t offset_)
: node_(node_),
  offset_(offset_),
  unique_(node_->graph_->next_unique_++),
  stage_(node_->graph_->new_node_stage_),
  has_unique_name_(false) {
  node_->graph_->all_values.emplace(this);
}

inline Graph * Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph * Value::owningGraph() const {
  return node()->owningGraph();
}

inline void Value::replaceAllUsesWith(Value * newValue) {
  ONNX_ASSERT(owningGraph() == newValue->owningGraph());
  for(auto u : uses()) {
    u.user->inputs_[u.offset] = newValue;
    newValue->uses_.push_back(u);
  }
  uses_.clear();
}

inline Node::Node(Graph * graph_, NodeKind kind_) :
  kind_(kind_),
  graph_(graph_),
  stage_(graph_->new_node_stage_),
  has_name_(false),
  has_doc_string_(false) {
  graph_->all_nodes.emplace(this);
}

inline void Node::eraseOutput(size_t i) {
  ONNX_ASSERT(i < outputs_.size());
  ONNX_ASSERT(outputs_[i]->uses().size() == 0);
  Value * n = outputs_[i];
  outputs_.erase(outputs_.begin() + i);
  owningGraph()->freeValue(n);
  for(size_t j = i; j < outputs_.size(); j++) {
    outputs_[j]->offset_--;
  }
}

inline void Node::destroy() {
  ONNX_ASSERT(inGraphList());
  while(outputs().size() > 0)
    eraseOutput(outputs().size() - 1);
  removeAllInputs();
  removeFromList();
  graph_->freeNode(this);
}

/************* All nodes not required to be defined before Graph **************/

inline graph_node_list_iterator Node::iterator() {
  return graph_node_list_iterator(this, 0);
}
inline graph_node_list_iterator Node::reverseIterator() {
  return iterator().reverse();
}
inline const_graph_node_list_iterator Node::iterator() const {
  return const_graph_node_list_iterator(this, 0);
}
inline const_graph_node_list_iterator Node::reverseIterator() const {
  return iterator().reverse();
}

} // namespace ONNX_NAMESPACE
