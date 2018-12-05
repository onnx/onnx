// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "onnx/common/assertions.h"

namespace ONNX_NAMESPACE {

struct EnumHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// Optional Annotation of an OpSchema which provides a level of generalization
// to the optimization framework. These are simple properties of the Op in
// question. Some annotations hierarchically contain other annotations. For
// example an ElementwiseIndependent annotation necessarily contains
// ElementwiseAny. This hierarchy is expressed in the OpAnnotation class.
//
// If you're adding an extra flag please remember to add it to the registry
// below.
enum class OpAnnotationFlag : uint16_t {
  // Op is applied elementwise to the input. The output shape exactly matches
  // the input shape.
  ElementwiseAny,
  // ElementwiseIndependent means the elementwise application of the Op does not
  // rely on any other elements in that tensor. All unary mathematical offers
  // are like this (e.g. Log, Exp, Sqrt).
  ElementwiseIndependent,
  // ElementwiseDependent is also an elementwise, but the elementwise dependent
  // Op relies on information outside the current value. The softmax family of
  // operators is a good example of an ElementwiseDependent Op. Although softmax
  // is applied elementwise the partition portion of the calculation requires a
  // summation across axis, therefore utilizing information outside of the
  // current value.
  ElementwiseDependent,
  // Elementwise operation that is strictly monotonic increasing. In other words
  // op(x) > op(y) iff x > y
  ElementwiseStrictMonotonicIncreasing,
  // Elementwise operation that is strictly monotonic decreasing. In other words
  // op(x) < op(y) iff x > y
  ElementwiseStrictMonotonicDecreasing,
  // Elementwise operation that is weak monotonic increasing. In other words
  // op(x) >= op(y) iff x > y
  ElementwiseWeakMonotonicIncreasing,
  // Elementwise operation that is weak monotonic decreasing. In other words
  // op(x) <= op(y) iff x > y
  ElementwiseWeakMonotonicDecreasing,
  // Annotation denoting whether or not the Op is a reduction Op. Please check
  // out onnx/defs/reduction for more information.
  Reduction,
  // Annotation denoting that the tensor values are unchanged, while the shape
  // or view is changed. Op's such as transpose and reshape qualify, but expand
  // doesn't since it increases the number of elements in the tensor.
  ShapeTransform
};

using OpAnnotationFlagSet = std::unordered_set<OpAnnotationFlag, EnumHash>;

class OpAnnotationHash;

// This class is a lightweight helper class over OpAnnotationFlag that allows us
// to build hierarchies within the flags. For example if an OP is strictly
// monotonic than it also satisfies the weak monotonic condition. So we can
// append both annotations to the Op, if the first flag is used but not the
// second.
class OpAnnotation {
  friend OpAnnotationHash;

 public:
  OpAnnotation(
      OpAnnotationFlag top_level_flag,
      std::shared_ptr<OpAnnotationFlagSet> annotations) {
    this->top_level_flag_ = top_level_flag;
    this->annotations_ = annotations;
    this->annotations_->insert(this->top_level_flag_);
  };

  OpAnnotation(OpAnnotationFlag top_level_flag)
      : OpAnnotation(
            top_level_flag,
            std::shared_ptr<OpAnnotationFlagSet>(new OpAnnotationFlagSet())) {}

  bool operator==(const OpAnnotation& other) const {
    return this->top_level_flag_ == other.top_level_flag_;
  }

  inline std::shared_ptr<OpAnnotationFlagSet> GetAnnotations() const {
    return this->annotations_;
  }

  inline OpAnnotationFlag GetTopLevelFlag() const {
    return this->top_level_flag_;
  }

 private:
  OpAnnotationFlag top_level_flag_;
  std::shared_ptr<OpAnnotationFlagSet> annotations_;
};

class OpAnnotationHash {
 public:
  size_t operator()(const OpAnnotation& x) const {
    return EnumHash()(x.top_level_flag_);
  }
};

class ElementwiseAnyOpAnn final : public OpAnnotation {
 public:
  ElementwiseAnyOpAnn() : OpAnnotation(OpAnnotationFlag::ElementwiseAny) {}
};
class ElementwiseIndependentOpAnn final : public OpAnnotation {
 public:
  ElementwiseIndependentOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseIndependent,
            std::shared_ptr<OpAnnotationFlagSet>(
                new OpAnnotationFlagSet{OpAnnotationFlag::ElementwiseAny})) {}
};
class ElementwiseDependentOpAnn final : public OpAnnotation {
 public:
  ElementwiseDependentOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseDependent,
            std::shared_ptr<OpAnnotationFlagSet>(
                new OpAnnotationFlagSet{OpAnnotationFlag::ElementwiseAny})) {}
};

class ElementwiseStrictMonotonicIncreasingOpAnn final : public OpAnnotation {
 public:
  ElementwiseStrictMonotonicIncreasingOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseStrictMonotonicIncreasing,
            std::shared_ptr<OpAnnotationFlagSet>(
                new OpAnnotationFlagSet{OpAnnotationFlag::ElementwiseAny})) {}
};
class ElementwiseWeakMonotonicIncreasingOpAnn final : public OpAnnotation {
 public:
  ElementwiseWeakMonotonicIncreasingOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseWeakMonotonicIncreasing,
            std::shared_ptr<OpAnnotationFlagSet>(new OpAnnotationFlagSet{
                OpAnnotationFlag::ElementwiseAny,
                OpAnnotationFlag::ElementwiseStrictMonotonicIncreasing})) {}
};
class ElementwiseStrictMonotonicDecreasingOpAnn final : public OpAnnotation {
 public:
  ElementwiseStrictMonotonicDecreasingOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseStrictMonotonicDecreasing,
            std::shared_ptr<OpAnnotationFlagSet>(
                new OpAnnotationFlagSet{OpAnnotationFlag::ElementwiseAny})) {}
};
class ElementwiseWeakMonotonicDecreasingOpAnn final : public OpAnnotation {
 public:
  ElementwiseWeakMonotonicDecreasingOpAnn()
      : OpAnnotation(
            OpAnnotationFlag::ElementwiseWeakMonotonicDecreasing,
            std::shared_ptr<OpAnnotationFlagSet>(new OpAnnotationFlagSet{
                OpAnnotationFlag::ElementwiseAny,
                OpAnnotationFlag::ElementwiseStrictMonotonicDecreasing})) {}
};
class ReductionOpAnn final : public OpAnnotation {
 public:
  ReductionOpAnn() : OpAnnotation(OpAnnotationFlag::Reduction) {}
};
class ShapeTransformOpAnn final : public OpAnnotation {
 public:
  ShapeTransformOpAnn() : OpAnnotation(OpAnnotationFlag::ShapeTransform) {}
};

using OpAnnotationRegistry_t = std::unordered_map<
    OpAnnotationFlag,
    const std::shared_ptr<OpAnnotation>,
    EnumHash>;

// Registry that contains a mapping from OpAnnotationFlag to OpAnnotation. This
// also the user to use flags and automaticaly reap the benefit of hierarchies
// of OpAnnotations provided by our library.
class OpAnnotationRegistry final {
 public:
  static std::shared_ptr<OpAnnotationRegistry> GetInstance() {
    return OpAnnotationRegistry::instance_;
  }
  // We work under the assumption that all flags are available in our registry.
  // We fail if this is not the case.
  std::shared_ptr<OpAnnotation> GetOpAnnotation(
      const OpAnnotationFlag flag) const {
    auto element = this->op_flag_mapping_->find(flag);
    ONNX_ASSERTM(
        element != this->op_flag_mapping_->end(),
        "Annotation is not found in registry. Please add your flag to the registry.");
    return element->second;
  }

 private:
  OpAnnotationRegistry() {
    RegisterAnnotation<ElementwiseAnyOpAnn>(OpAnnotationFlag::ElementwiseAny);
    RegisterAnnotation<ElementwiseIndependentOpAnn>(
        OpAnnotationFlag::ElementwiseIndependent);
    RegisterAnnotation<ElementwiseDependentOpAnn>(
        OpAnnotationFlag::ElementwiseDependent);
    RegisterAnnotation<ElementwiseStrictMonotonicIncreasingOpAnn>(
        OpAnnotationFlag::ElementwiseStrictMonotonicIncreasing);
    RegisterAnnotation<ElementwiseStrictMonotonicDecreasingOpAnn>(
        OpAnnotationFlag::ElementwiseStrictMonotonicDecreasing);
    RegisterAnnotation<ElementwiseWeakMonotonicIncreasingOpAnn>(
        OpAnnotationFlag::ElementwiseWeakMonotonicIncreasing);
    RegisterAnnotation<ElementwiseWeakMonotonicDecreasingOpAnn>(
        OpAnnotationFlag::ElementwiseWeakMonotonicDecreasing);
    RegisterAnnotation<ReductionOpAnn>(OpAnnotationFlag::Reduction);
    RegisterAnnotation<ShapeTransformOpAnn>(OpAnnotationFlag::ShapeTransform);
  }

  template <typename T>
  void RegisterAnnotation(OpAnnotationFlag flag) {
    static_assert(
        std::is_base_of<OpAnnotation, T>::value,
        "T must inherit from OpAnnotation");
    auto annotation = std::shared_ptr<OpAnnotation>(new T());
    // Top level flags must match
    ONNX_ASSERT(annotation->GetTopLevelFlag() == flag);
    bool inserted = this->op_flag_mapping_->insert({flag, annotation}).second;
    // Repeated elements cannot occur in OpAnnotationRegistry
    ONNX_ASSERT(inserted);
  }
  std::unique_ptr<OpAnnotationRegistry_t> op_flag_mapping_ =
      std::unique_ptr<OpAnnotationRegistry_t>(new OpAnnotationRegistry_t());
  static std::shared_ptr<OpAnnotationRegistry> instance_;
};
} // namespace ONNX_NAMESPACE
