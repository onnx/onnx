// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <string>
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct PostPassAnalysis {};

enum PassType { Fuse = 0, Nop = 1, Seperate = 2, Immutable = 3 };
enum PassEfficiency { Partial = 0, Complete = 1 };
enum PassOptimizationType {
  None = 0,
  Compute = 1,
  Memory = 2,
  ComputeMemory = 3,
  Stability = 4
};

class Pass {
  PassType pass_type;
  PassEfficiency pass_efficiency;
  PassOptimizationType pass_optimization_type;

 public:
  explicit Pass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
      : pass_type(pass_type),
        pass_efficiency(pass_efficiency),
        pass_optimization_type(pass_optimization_type) {}
  virtual ~Pass();

  PassType getPassType() const {
    return this->pass_type;
  }
  PassEfficiency getPassEfficiency() const {
    return this->pass_efficiency;
  }
  PassOptimizationType getPassOptimizationType() const {
    return this->pass_optimization_type;
  }
  virtual std::string getPassName() const;

  virtual bool initializePass(Graph&) {
    return false;
  }
  virtual bool finalizePass(Graph&) {
    return false;
  }
  virtual PostPassAnalysis runPass(Graph& graph) = 0;
  virtual PostPassAnalysis runPass(ONNX_NAMESPACE::ModelProto& graph) = 0;

 protected:
  uint DescendOnGraphAttributes(Node* n, std::function<uint(Graph&)> fn);
};

class ImmutablePass : Pass {
 public:
  explicit ImmutablePass()
      : Pass(
            PassType::Immutable,
            PassEfficiency::Complete,
            PassOptimizationType::None) {}
  ~ImmutablePass() override;
};

struct PostPredictBasedPassAnalysis : PostPassAnalysis {
  Pass* pass;
  uint num_positive_transforms;
  bool succesful_initialization;
  bool succesful_finalization;

 public:
  explicit PostPredictBasedPassAnalysis(
      Pass* pass,
      uint num_positive_transforms,
      bool succesful_initialization,
      bool succesful_finalization)
      : pass(pass),
        num_positive_transforms(num_positive_transforms),
        succesful_initialization(succesful_initialization),
        succesful_finalization(succesful_finalization) {}

  bool graphChanged() {
    return this->num_positive_transforms > 0;
  }
  bool numSucceededTransforms() {
    return this->num_positive_transforms;
  }
  bool fixedPointOptimizationNeeded() {
    return this->graphChanged() &&
        pass->getPassEfficiency() == PassEfficiency::Partial;
  }
};

class PredicateBasedPass final : Pass {
 public:
  explicit PredicateBasedPass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
      : Pass(pass_type, pass_efficiency, pass_optimization_type) {}
  ~PredicateBasedPass() override;

  virtual bool patternMatchPredicate(Node* node) {
    return false;
  }
  virtual bool runTransform(Node* node) {
    return false;
  }

  PostPassAnalysis runPass(Graph& graph);
  PostPassAnalysis runPass(ONNX_NAMESPACE::ModelProto& graph);

 private:
  uint _runPassInternal(Graph& graph);
  uint _runPassInternal(ONNX_NAMESPACE::ModelProto& graph);
};

class FullGraphBasedPass : Pass {
 public:
  explicit FullGraphBasedPass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
      : Pass(pass_type, pass_efficiency, pass_optimization_type) {}
  ~FullGraphBasedPass() override;
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
