// Default converter for ONNX models between different opset versions
// in the default domain ("" or "ai.onnx").

#pragma once

#include "onnx/version_converter/BaseConverter.h"
#include "onnx/version_converter/adapters/argmax_argmin_12_11.h"
#include "onnx/version_converter/adapters/averagepool_7_6.h"
#include "onnx/version_converter/adapters/axes_attribute_to_input.h"
#include "onnx/version_converter/adapters/axes_input_to_attribute.h"
#include "onnx/version_converter/adapters/batch_normalization_6_5.h"
#include "onnx/version_converter/adapters/batch_normalization_6_7.h"
#include "onnx/version_converter/adapters/batch_normalization_8_9.h"
#include "onnx/version_converter/adapters/broadcast_backward_compatibility.h"
#include "onnx/version_converter/adapters/broadcast_forward_compatibility.h"
#include "onnx/version_converter/adapters/cast_9_8.h"
#include "onnx/version_converter/adapters/clip_10_11.h"
#include "onnx/version_converter/adapters/compatible.h"
#include "onnx/version_converter/adapters/concat_3_4.h"
#include "onnx/version_converter/adapters/dropout_11_12.h"
#include "onnx/version_converter/adapters/dropout_6_7.h"
#include "onnx/version_converter/adapters/extend_supported_types.h"
#include "onnx/version_converter/adapters/gemm_6_7.h"
#include "onnx/version_converter/adapters/gemm_7_6.h"
#include "onnx/version_converter/adapters/maxpool_8_7.h"
#include "onnx/version_converter/adapters/no_previous_version.h"
#include "onnx/version_converter/adapters/remove_consumed_inputs.h"
#include "onnx/version_converter/adapters/reshape_4_5.h"
#include "onnx/version_converter/adapters/reshape_5_4.h"
#include "onnx/version_converter/adapters/scan_8_9.h"
#include "onnx/version_converter/adapters/scan_9_8.h"
#include "onnx/version_converter/adapters/set_is_test.h"
#include "onnx/version_converter/adapters/split_12_13.h"
#include "onnx/version_converter/adapters/split_13_12.h"
#include "onnx/version_converter/adapters/sum_8_7.h"
#include "onnx/version_converter/adapters/slice_9_10.h"
#include "onnx/version_converter/adapters/type_restriction.h"
#include "onnx/version_converter/adapters/upsample_8_9.h"
#include "onnx/version_converter/adapters/upsample_9_8.h"

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

      /******** 3 -> 4 ********/
      registerAdapter(make_unique<Concat_3_4>());

      /******** 4 -> 3 ********/
      std::vector<TensorProto_DataType> concat_unallowed_types = {
        TensorProto_DataType_INT32, TensorProto_DataType_INT64,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64,
        TensorProto_DataType_UINT8, TensorProto_DataType_UINT16,
        TensorProto_DataType_INT8, TensorProto_DataType_INT16,
        TensorProto_DataType_STRING, TensorProto_DataType_BOOL};
      registerAdapter(make_unique<TypeRestriction>("Concat", OpSetID(4),
            OpSetID(3), concat_unallowed_types));

      /******** 4 -> 5 ********/
      registerAdapter(make_unique<Reshape_4_5>());

      /******** 5 -> 4 ********/
      registerAdapter(make_unique<Reshape_5_4>());

      /******** 5 -> 6 ********/
      registerAdapter(make_unique<RemoveConsumedInputs>("Add",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Mul",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Relu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("BatchNormalization",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Sum",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Dropout",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Abs",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Ceil",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Clip",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Div",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Elu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Exp",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Floor",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("HardSigmoid",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("InstanceNormalization",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("LeakyRelu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Log",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Max",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Mean",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Min",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Neg",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("PRelu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Reciprocal",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Selu",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Sigmoid",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Sqrt",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Sub",
        OpSetID(5), OpSetID(6)));
      registerAdapter(make_unique<RemoveConsumedInputs>("Tanh",
        OpSetID(5), OpSetID(6)));

      /******** 6 -> 5 ********/
      std::vector<TensorProto_DataType> broadcast_unallowed_types = {
        TensorProto_DataType_INT32, TensorProto_DataType_INT64,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64};
      std::vector<TensorProto_DataType> int_unallowed_types = {
        TensorProto_DataType_UINT8, TensorProto_DataType_UINT16,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64,
        TensorProto_DataType_INT8, TensorProto_DataType_INT16,
        TensorProto_DataType_INT32, TensorProto_DataType_INT64};
      std::vector<TensorProto_DataType> neg_unallowed_types = {
        TensorProto_DataType_INT32, TensorProto_DataType_INT8,
        TensorProto_DataType_UINT16, TensorProto_DataType_INT64};
      registerAdapter(make_unique<TypeRestriction>("Add",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Mul",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Sub",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Div",
        OpSetID(6), OpSetID(5), broadcast_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Abs",
        OpSetID(6), OpSetID(5), int_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Neg",
        OpSetID(6), OpSetID(5), neg_unallowed_types));
      registerAdapter(make_unique<BatchNormalization_6_5>());
registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(6), OpSetID(5)));
registerAdapter(make_unique<CompatibleAdapter>("Relu",
        OpSetID(6), OpSetID(5)));
registerAdapter(make_unique<CompatibleAdapter>("Sum",
        OpSetID(6), OpSetID(5)));
registerAdapter(make_unique<CompatibleAdapter>("Dropout",
        OpSetID(6), OpSetID(5)));
      /******** 6 -> 7 ********/
      registerAdapter(make_unique<CompatibleAdapter>("AveragePool",
        OpSetID(6), OpSetID(7)));      
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Add",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Div",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Mul",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Pow",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<CompatibleAdapter>("PRelu",
        OpSetID(6), OpSetID(7)));    
      registerAdapter(make_unique<BroadcastForwardCompatibility>("Sub",
        OpSetID(6), OpSetID(7)));
      registerAdapter(make_unique<Gemm_6_7>());
      registerAdapter(make_unique<BatchNormalization_6_7>());
      registerAdapter(make_unique<Dropout_6_7>());

      /******** 7 -> 6 ********/   
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Add",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Div",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Mul",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Pow",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<CompatibleAdapter>("PRelu",
        OpSetID(7), OpSetID(6)));  
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Sub",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<SetIsTest>("BatchNormalization",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<SetIsTest>("Dropout",
        OpSetID(7), OpSetID(6)));
      registerAdapter(make_unique<Gemm_7_6>());
      registerAdapter(make_unique<AveragePool_7_6>());

      /******** 7 -> 8 ********/  
      registerAdapter(make_unique<CompatibleAdapter>("Max",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<CompatibleAdapter>("Min",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<CompatibleAdapter>("Mean",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<CompatibleAdapter>("Sum",
        OpSetID(7), OpSetID(8)));
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(7), OpSetID(8)));

      /******** 8 -> 7 ********/  
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Max",
        OpSetID(8), OpSetID(7)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Min",
        OpSetID(8), OpSetID(7)));
      registerAdapter(make_unique<BroadcastBackwardCompatibility>("Mean",
        OpSetID(8), OpSetID(7)));      
      registerAdapter(make_unique<Sum_8_7>());
      registerAdapter(make_unique<MaxPool_8_7>());

      /******** 8 -> 9 ********/  
      registerAdapter(make_unique<CompatibleAdapter>("Flatten",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("Constant",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("MatMul",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("PRelu",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("Greater",
        OpSetID(8), OpSetID(9)));      
      registerAdapter(make_unique<CompatibleAdapter>("Less",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<CompatibleAdapter>("Cast",
        OpSetID(8), OpSetID(9)));
      registerAdapter(make_unique<BatchNormalization_8_9>());
      registerAdapter(make_unique<Scan_8_9>());
      registerAdapter(make_unique<Upsample_8_9>());

      /******** 9 -> 8 ********/
      registerAdapter(make_unique<CompatibleAdapter>("BatchNormalization",
        OpSetID(9), OpSetID(8)));  
      registerAdapter(make_unique<ExtendSupportedTypes>("Flatten",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("Constant",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("MatMul",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("Gemm",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("PRelu",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("Greater",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<ExtendSupportedTypes>("Less",
        OpSetID(9), OpSetID(8)));
      registerAdapter(make_unique<Cast_9_8>());
      registerAdapter(make_unique<Scan_9_8>());
      registerAdapter(make_unique<Upsample_9_8>());

      /******** 9 -> 10 ********/
      registerAdapter(make_unique<CompatibleAdapter>("AveragePool",
        OpSetID(9), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(9), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Dropout",
        OpSetID(9), OpSetID(10)));
      registerAdapter(make_unique<Slice_9_10>());

      /******** 10 -> 9 ********/
      registerAdapter(make_unique<CompatibleAdapter>("Dropout",
        OpSetID(10), OpSetID(9)));

      /******** 10 -> 11 ********/
      registerAdapter(make_unique<CompatibleAdapter>("ArgMax",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ArgMin",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("AveragePool",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Concat",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Constant",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Compress",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Conv",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ConvTranspose",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Equal",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Flatten",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Gather",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Gemm",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Hardmax",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("LogSoftmax",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceL1",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceL2",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceLogSum",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceLogSumExp",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMax",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMean",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMin",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceProd",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceSum",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceSumSquare",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Scan",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Softmax",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Slice",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Squeeze",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<CompatibleAdapter>("Unsqueeze",
        OpSetID(10), OpSetID(11)));
      registerAdapter(make_unique<Clip_10_11>());

      /******** 11 -> 10 ********/
      std::vector<TensorProto_DataType> equal_unallowed_types = {
        TensorProto_DataType_UINT8, TensorProto_DataType_UINT16,
        TensorProto_DataType_UINT32, TensorProto_DataType_UINT64,
        TensorProto_DataType_INT8, TensorProto_DataType_INT16,
        TensorProto_DataType_FLOAT16, TensorProto_DataType_FLOAT,
        TensorProto_DataType_DOUBLE};
      registerAdapter(make_unique<CompatibleAdapter>("ArgMax",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ArgMin",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("AveragePool",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Concat",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Constant",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Conv",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ConvTranspose",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<TypeRestriction>("Equal",
        OpSetID(11), OpSetID(10), equal_unallowed_types));
      registerAdapter(make_unique<CompatibleAdapter>("Flatten",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("LogSoftmax",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceL1",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceL2",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceLogSum",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceLogSumExp",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMax",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMean",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMin",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceProd",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceSum",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceSumSquare",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Softmax",
        OpSetID(11), OpSetID(10)));
      registerAdapter(make_unique<CompatibleAdapter>("Unsqueeze",
        OpSetID(11), OpSetID(10)));

      /******** 11 -> 12 ********/
      registerAdapter(make_unique<CompatibleAdapter>("ArgMax",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("ArgMin",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("BatchNormalization",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("Constant",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("Clip",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("Min",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("Max",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("MaxPool",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("Pow",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMax",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<CompatibleAdapter>("ReduceMin",
        OpSetID(11), OpSetID(12)));
      registerAdapter(make_unique<Dropout_11_12>());

      /******** 12 -> 11 ********/
      std::vector<TensorProto_DataType> maxpool_unallowed_types = {
        TensorProto_DataType_UINT8, TensorProto_DataType_INT8};
      registerAdapter(make_unique<ArgMaxArgMin_12_11>("ArgMax"));
      registerAdapter(make_unique<ArgMaxArgMin_12_11>("ArgMin"));
      registerAdapter(make_unique<CompatibleAdapter>("BatchNormalization",
        OpSetID(12), OpSetID(11)));
      registerAdapter(make_unique<TypeRestriction>("Clip", 
        OpSetID(12), OpSetID(11), int_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Min", 
        OpSetID(12), OpSetID(11), int_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("Max", 
        OpSetID(12), OpSetID(11), int_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("MaxPool", 
        OpSetID(12), OpSetID(11), maxpool_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("ReduceMax", 
        OpSetID(12), OpSetID(11), maxpool_unallowed_types));
      registerAdapter(make_unique<TypeRestriction>("ReduceMin", 
        OpSetID(12), OpSetID(11), maxpool_unallowed_types));

      /******** 12 -> 13 ********/
      registerAdapter(make_unique<CompatibleAdapter>(
          "Constant", OpSetID(12), OpSetID(13)));
      registerAdapter(make_unique<AxesAttributeToInput>(
          "ReduceSum", OpSetID(12), OpSetID(13)));
      registerAdapter(make_unique<AxesAttributeToInput>(
          "Squeeze", OpSetID(12), OpSetID(13)));
      registerAdapter(make_unique<AxesAttributeToInput>(
          "Unsqueeze", OpSetID(12), OpSetID(13)));
      registerAdapter(make_unique<Split_12_13>());

      /******** 13 -> 12 ********/
      registerAdapter(make_unique<CompatibleAdapter>(
          "Constant", OpSetID(13), OpSetID(12)));
      registerAdapter(make_unique<AxesInputToAttribute>(
          "ReduceSum", OpSetID(13), OpSetID(12)));
      registerAdapter(make_unique<AxesInputToAttribute>(
          "Squeeze", OpSetID(13), OpSetID(12)));
      registerAdapter(make_unique<AxesInputToAttribute>(
          "Unsqueeze", OpSetID(13), OpSetID(12)));
      registerAdapter(make_unique<Split_13_12>());
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
