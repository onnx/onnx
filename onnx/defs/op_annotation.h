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

enum OpAnnotationFlag : uint16_t {
  ElementwiseAny,
  ElementwiseIndependent,
  ElementwiseDependent,
  ElementwiseStrictMonotonicIncreasing,
  ElementwiseStrictMonotonicDecreasing,
  ElementwiseWeakMonotonicIncreasing,
  ElementwiseWeakMonotonicDecreasing,
  Reduction,
  ShapeTransform
};

using OpAnnotationFlagSet = std::unordered_set<OpAnnotationFlag, EnumHash>;

class OpAnnotationHash;

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
    return std::hash<uint16_t>()(x.top_level_flag_);
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

class OpAnnotationRegistry final {
 public:
  static std::shared_ptr<OpAnnotationRegistry> GetInstance() {
    return OpAnnotationRegistry::instance_;
  }
  std::shared_ptr<OpAnnotation> GetOpAnnotation(
      const OpAnnotationFlag flag) const {
    auto element = this->op_flag_mapping_->find(flag);
    ONNX_ASSERTM(
        element != this->op_flag_mapping_->end(),
        "Annotation is not found in registry");
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
