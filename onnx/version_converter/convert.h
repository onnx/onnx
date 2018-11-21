// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/no_previous_version.h"
#include "onnx/version_converter/adapters/broadcast_backward_compatibility.h"
#include "onnx/version_converter/adapters/broadcast_forward_compatibility.h"
#include "onnx/version_converter/adapters/type_restriction.h"
#include "onnx/version_converter/adapters/compatible.h"
#include "onnx/version_converter/adapters/remove_consumed_inputs.h"
#include "onnx/version_converter/adapters/gemm_7_6.h"
#include "onnx/version_converter/adapters/gemm_6_7.h"
#include "onnx/version_converter/adapters/batch_normalization_6_5.h"
#include "onnx/version_converter/adapters/batch_normalization_6_7.h"
#include "onnx/version_converter/adapters/set_is_test.h"
#include "onnx/version_converter/adapters/concat_3_4.h"
#include "onnx/version_converter/adapters/reshape_5_4.h"
#include "onnx/version_converter/adapters/reshape_4_5.h"
#include "onnx/version_converter/adapters/sum_8_7.h"
#include "onnx/version_converter/adapters/averagepool_7_6.h"
#include "onnx/version_converter/adapters/dropout_6_7.h"
#include "onnx/version_converter/adapters/maxpool_8_7.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class DefaultVersionConverter : public BaseVersionConverter {
  private:
    bool DEBUG = false;

    std::pair<int, int> version_range;

    bool searchOpDomainMap(const std::unordered_map<std::string, std::map<
      int64_t, const OpSchema*>>& op_domain_map, int64_t curr_version,
      int64_t step) const {
      bool up = step == 1;
      const auto version_it = op_domain_map.find("");
      return version_it != op_domain_map.end() &&
          ((version_it->second.find(curr_version) !=
          version_it->second.end() && !up) ||
          (version_it->second.find(curr_version + step) !=
          version_it->second.end() && up));
    }

    void debug(const std::string& str) const {
      if (DEBUG) std::cerr << str << std::endl;
    }

    void assertInVersionRange(int64_t version) const {
      ONNX_ASSERTM(version >= version_range.first && version <=
          version_range.second,
          "Warning: invalid version (must be between %s and %s)",
          version_range.first, version_range.second);
    }

    void assertDefaultDomain(const std::string& initial_domain,
        const std::string& target_domain) const {
      ONNX_ASSERTM((initial_domain == "" || initial_domain == "ai.onnx") &&
          (target_domain == "" || target_domain == "ai.onnx"),
          "Warning: default onnx version converter can only convert "
          " between default domain opset versions ('' or 'ai.onnx')\n");
      ONNX_ASSERTM(initial_domain == target_domain,
          "initial_version and target_version must have the same domains");
    }

  public:
    DefaultVersionConverter() {
      const std::unordered_map<std::string, std::pair<int, int>>& versions_map =
        OpSchemaRegistry::DomainToVersionRange::Instance().Map();
      version_range = versions_map.at("");
      // Register adapters to the version converter
      const std::vector<OpSchema> all_opschemas =
        OpSchemaRegistry::get_all_schemas_with_history();

      for (const OpSchema& schema : all_opschemas) {
        all_schemas[schema.Name()][schema.domain()][(int64_t)
          schema.since_version()] = &schema;
      }

      // Iterate through all_schemas to determine NoPreviousVersionAdapters
      for (auto& op_pair : all_schemas) {
        const auto default_versions = op_pair.second.find("");
        if (default_versions != op_pair.second.end()) {
          int64_t min_version = version_range.second;
          for (auto& version_pair : default_versions->second) {
            if (version_pair.first < min_version) {
              min_version = version_pair.first;
            }
          }
          if (min_version > 1) {
            registerAdapter(make_unique<NoPreviousVersionAdapter>(op_pair.first,
              OpSetID(min_version), OpSetID(min_version - 1)));
          }
        }
      }

      std::vector<TensorProto_DataType> broadcast_unallowed_types = {
        TensorProto_DataType_INT32, TensorProto_DataType_INT64,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64};
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Add",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Add",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<TypeRestriction>("Add",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<RemoveConsumedInputs>("Add",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Mul",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Mul",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<TypeRestriction>("Mul",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<RemoveConsumedInputs>("Mul",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<Gemm_7_6>());
      registerAdapter(make_unique<Gemm_6_7>());
      registerAdapter(make_unique<RemoveConsumedInputs>("Relu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<CompatibleAdapter>("Relu",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<SetIsTest>("BatchNormalization",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BatchNormalization_6_7>());
      registerAdapter(make_unique<BatchNormalization_6_5>());
      registerAdapter(make_unique<RemoveConsumedInputs>("BatchNormalization",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<Concat_3_4>());
      std::vector<TensorProto_DataType> concat_unallowed_types = {
        TensorProto_DataType_INT32, TensorProto_DataType_INT64,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64,
        TensorProto_DataType_UINT8, TensorProto_DataType_UINT16,
        TensorProto_DataType_INT8, TensorProto_DataType_INT16,
        TensorProto_DataType_STRING, TensorProto_DataType_BOOL};
      registerAdapter(make_unique<TypeRestriction>("Concat", OpSetID(4),
            OpSetID(3), concat_unallowed_types));
      registerAdapter(make_unique<Reshape_4_5>());
      registerAdapter(make_unique<Reshape_5_4>());
      registerAdapter(make_unique<CompatibleAdapter>("Sum",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<CompatibleAdapter>("Sum",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Sum",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<Sum_8_7>());
      registerAdapter(make_unique<CompatibleAdapter>("Dropout",
        OpSetID(6), OpSetID(5)));
      registerAdapter(make_unique<SetIsTest>("Dropout",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Dropout",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<Dropout_6_7>());
      registerAdapter(make_unique<CompatibleAdapter>("AveragePool",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<AveragePool_7_6>());
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<MaxPool_8_7>());
    }

    ModelProto convert_version(
        const ModelProto& mp_in,
        const OpSetID& initial_version,
        const OpSetID& target_version) const override;
};

ModelProto ConvertVersion(
    const ModelProto& mp_in,
    int target_version);
}} // namespace ONNX_NAMESPACE::version_conversion
