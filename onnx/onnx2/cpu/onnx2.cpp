#include "onnx2.h"
#include "stream_class.hpp"
#include <sstream>

namespace onnx2 {

// StringStringEntryProto

IMPLEMENT_PROTO(StringStringEntryProto)
uint64_t StringStringEntryProto::SerializeSize(utils::BinaryWriteStream &stream,
                                               SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, key)
  SIZE_FIELD(size, options, stream, value)
  return size;
}
void StringStringEntryProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                               SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key)
  WRITE_FIELD(options, stream, value)
}
void StringStringEntryProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, StringStringEntryProto) //
    READ_FIELD(options, stream, key)                    //
    READ_FIELD(options, stream, value)                  //
    READ_END(options, stream, StringStringEntryProto)   //  // NOLINT
} std::vector<std::string> StringStringEntryProto::PrintToVectorString(utils::PrintOptions &options)
    const {
  return {write_as_string(options, NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value))};
}

// TensorAnnotation
IMPLEMENT_PROTO(TensorAnnotation)
uint64_t TensorAnnotation::SerializeSize(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, tensor_name)
  SIZE_REPEATED_FIELD(size, options, stream, quant_parameter_tensor_names)
  return size;
}
void TensorAnnotation::SerializeToStream(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  WRITE_FIELD(options, stream, tensor_name)
  WRITE_REPEATED_FIELD(options, stream, quant_parameter_tensor_names)
}
void TensorAnnotation::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TensorAnnotation)                      //
    READ_FIELD(options, stream, tensor_name)                           //
    READ_REPEATED_FIELD(options, stream, quant_parameter_tensor_names) //
    READ_END(options, stream, TensorAnnotation)                        //  // NOLINT
} std::vector<std::string> TensorAnnotation::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(tensor_name),
                                        NAME_EXIST_VALUE(quant_parameter_tensor_names));
}

// IntIntListEntryProto

IMPLEMENT_PROTO(IntIntListEntryProto)
uint64_t IntIntListEntryProto::SerializeSize(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, key)
  SIZE_REPEATED_FIELD(size, options, stream, value)
  return size;
}
void IntIntListEntryProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key)
  WRITE_REPEATED_FIELD(options, stream, value)
}
void IntIntListEntryProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, IntIntListEntryProto) //
    READ_FIELD(options, stream, key)                  //
    READ_REPEATED_FIELD(options, stream, value)       //
    READ_END(options, stream, IntIntListEntryProto)   //
} std::vector<std::string> IntIntListEntryProto::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(key), NAME_EXIST_VALUE(value));
}

// DeviceConfigurationProto

IMPLEMENT_PROTO(DeviceConfigurationProto)
uint64_t DeviceConfigurationProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, num_devices)
  SIZE_REPEATED_FIELD(size, options, stream, device)
  return size;
}
void DeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                                 SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, num_devices)
  WRITE_REPEATED_FIELD(options, stream, device)
}
void DeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, DeviceConfigurationProto) //
    READ_FIELD(options, stream, name)                     //
    READ_FIELD(options, stream, num_devices)              //
    READ_REPEATED_FIELD(options, stream, device)          //
    READ_END(options, stream, DeviceConfigurationProto)   //
} std::vector<std::string> DeviceConfigurationProto::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(num_devices),
                                        NAME_EXIST_VALUE(device));
}

// SimpleShardedDimProto

IMPLEMENT_PROTO(SimpleShardedDimProto)
uint64_t SimpleShardedDimProto::SerializeSize(utils::BinaryWriteStream &stream,
                                              SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, dim_value)
  SIZE_FIELD(size, options, stream, dim_param)
  SIZE_FIELD(size, options, stream, num_shards)
  return size;
}
void SimpleShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                              SerializeOptions &options) const {
  WRITE_FIELD(options, stream, dim_value)
  WRITE_FIELD(options, stream, dim_param)
  WRITE_FIELD(options, stream, num_shards)
}
void SimpleShardedDimProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, SimpleShardedDimProto) //
    READ_FIELD(options, stream, dim_value)             //
    READ_FIELD(options, stream, dim_param)             //
    READ_FIELD(options, stream, num_shards)            //
    READ_END(options, stream, SimpleShardedDimProto)   //
} std::vector<std::string> SimpleShardedDimProto::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(dim_value),
                                        NAME_EXIST_VALUE(dim_param), NAME_EXIST_VALUE(num_shards));
}

// ShardedDimProto

IMPLEMENT_PROTO(ShardedDimProto)
uint64_t ShardedDimProto::SerializeSize(utils::BinaryWriteStream &stream,
                                        SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, axis)
  SIZE_REPEATED_FIELD(size, options, stream, simple_sharding)
  return size;
}
void ShardedDimProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                        SerializeOptions &options) const {
  WRITE_FIELD(options, stream, axis)
  WRITE_REPEATED_FIELD(options, stream, simple_sharding)
}

void ShardedDimProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, ShardedDimProto)          //
    READ_FIELD(options, stream, axis)                     //
    READ_REPEATED_FIELD(options, stream, simple_sharding) //
    READ_END(options, stream, ShardedDimProto)            //
} std::vector<std::string> ShardedDimProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(axis),
                                        NAME_EXIST_VALUE(simple_sharding));
}

// ShardingSpecProto

IMPLEMENT_PROTO(ShardingSpecProto)
uint64_t ShardingSpecProto::SerializeSize(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, tensor_name)
  SIZE_REPEATED_FIELD(size, options, stream, device)
  SIZE_REPEATED_FIELD(size, options, stream, index_to_device_group_map)
  SIZE_REPEATED_FIELD(size, options, stream, sharded_dim)
  return size;
}
void ShardingSpecProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, tensor_name)
  WRITE_REPEATED_FIELD(options, stream, device)
  WRITE_REPEATED_FIELD(options, stream, index_to_device_group_map)
  WRITE_REPEATED_FIELD(options, stream, sharded_dim)
}
void ShardingSpecProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, ShardingSpecProto)                  //
    READ_FIELD(options, stream, tensor_name)                        //
    READ_REPEATED_FIELD(options, stream, device)                    //
    READ_REPEATED_FIELD(options, stream, index_to_device_group_map) //
    READ_REPEATED_FIELD(options, stream, sharded_dim)               //
    READ_END(options, stream, ShardingSpecProto)                    //  // NOLINT
} std::vector<std::string> ShardingSpecProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(tensor_name), NAME_EXIST_VALUE(device),
      NAME_EXIST_VALUE(index_to_device_group_map), NAME_EXIST_VALUE(sharded_dim));
}

// NodeDeviceConfigurationProto

IMPLEMENT_PROTO(NodeDeviceConfigurationProto)
uint64_t NodeDeviceConfigurationProto::SerializeSize(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, configuration_id)
  SIZE_REPEATED_FIELD(size, options, stream, sharding_spec)
  SIZE_FIELD(size, options, stream, pipeline_stage)
  return size;
}
void NodeDeviceConfigurationProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                                     SerializeOptions &options) const {
  WRITE_FIELD(options, stream, configuration_id)
  WRITE_REPEATED_FIELD(options, stream, sharding_spec)
  WRITE_FIELD(options, stream, pipeline_stage)
}
void NodeDeviceConfigurationProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, NodeDeviceConfigurationProto) //
    READ_FIELD(options, stream, configuration_id)             //
    READ_REPEATED_FIELD(options, stream, sharding_spec)       //
    READ_FIELD(options, stream, pipeline_stage)               //
    READ_END(options, stream, NodeDeviceConfigurationProto)   //
} std::vector<std::string> NodeDeviceConfigurationProto::PrintToVectorString(utils::PrintOptions
                                                                                 &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(configuration_id),
                                        NAME_EXIST_VALUE(sharding_spec),
                                        NAME_EXIST_VALUE(pipeline_stage));
}

// OperatorSetIdProto

IMPLEMENT_PROTO(OperatorSetIdProto)
uint64_t OperatorSetIdProto::SerializeSize(utils::BinaryWriteStream &stream,
                                           SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD_EMPTY(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, version)
  return size;
}
void OperatorSetIdProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                           SerializeOptions &options) const {
  WRITE_FIELD_EMPTY(options, stream, domain)
  WRITE_FIELD(options, stream, version)
}
void OperatorSetIdProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, OperatorSetIdProto) //
    READ_FIELD(options, stream, domain)             //
    READ_FIELD(options, stream, version)            //
    READ_END(options, stream, OperatorSetIdProto)   //
} std::vector<std::string> OperatorSetIdProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(version));
}

// TensorShapeProto::Dimension

IMPLEMENT_PROTO(TensorShapeProto::Dimension)
uint64_t TensorShapeProto::Dimension::SerializeSize(utils::BinaryWriteStream &stream,
                                                    SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, dim_value)
  SIZE_FIELD(size, options, stream, dim_param)
  SIZE_FIELD(size, options, stream, denotation)
  return size;
}
void TensorShapeProto::Dimension::SerializeToStream(utils::BinaryWriteStream &stream,
                                                    SerializeOptions &options) const {
  WRITE_FIELD(options, stream, dim_value)
  WRITE_FIELD(options, stream, dim_param)
  WRITE_FIELD(options, stream, denotation)
}
void TensorShapeProto::Dimension::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TensorShapeProto::Dimension) //
    READ_FIELD(options, stream, dim_value)                   //
    READ_FIELD(options, stream, dim_param)                   //
    READ_FIELD(options, stream, denotation)                  //
    READ_END(options, stream, TensorShapeProto::Dimension)   //
} std::vector<std::string> TensorShapeProto::Dimension::PrintToVectorString(utils::PrintOptions
                                                                                &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(dim_value),
                                        NAME_EXIST_VALUE(dim_param), NAME_EXIST_VALUE(denotation));
}

// TensorShapeProto

IMPLEMENT_PROTO(TensorShapeProto)
uint64_t TensorShapeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_REPEATED_FIELD(size, options, stream, dim)
  return size;
}
void TensorShapeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                         SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, dim)
}
void TensorShapeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TensorShapeProto) //
    READ_REPEATED_FIELD(options, stream, dim)     //
    READ_END(options, stream, TensorShapeProto)   //
} std::vector<std::string> TensorShapeProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(dim));
}

// TensorProto::Segment

IMPLEMENT_PROTO(TensorProto::Segment)
uint64_t TensorProto::Segment::SerializeSize(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, begin)
  SIZE_FIELD(size, options, stream, end)
  return size;
}
void TensorProto::Segment::SerializeToStream(utils::BinaryWriteStream &stream,
                                             SerializeOptions &options) const {
  WRITE_FIELD(options, stream, begin)
  WRITE_FIELD(options, stream, end)
}
void TensorProto::Segment::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TensorProto::Segment) //
    READ_FIELD(options, stream, begin)                //
    READ_FIELD(options, stream, end)                  //
    READ_END(options, stream, TensorProto::Segment)   //
} std::vector<std::string> TensorProto::Segment::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(begin), NAME_EXIST_VALUE(end));
}

// TensorProto

IMPLEMENT_PROTO(TensorProto)
uint64_t TensorProto::SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_REPEATED_FIELD(size, options, stream, dims)
  SIZE_ENUM_FIELD(size, options, stream, data_type)
  SIZE_ENUM_FIELD(size, options, stream, data_location)
  SIZE_FIELD_NULL(size, options, stream, name)
  SIZE_FIELD_LIMIT(size, options, stream, raw_data)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, external_data)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, double_data)
  SIZE_REPEATED_FIELD(size, options, stream, float_data)
  SIZE_REPEATED_FIELD(size, options, stream, int32_data)
  SIZE_REPEATED_FIELD(size, options, stream, int64_data)
  SIZE_REPEATED_FIELD(size, options, stream, uint64_data)
  SIZE_REPEATED_FIELD(size, options, stream, string_data)
  return size;
}
void TensorProto::SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  // Validation for external data.
  if (has_data_location() && ref_data_location() == DataLocation::EXTERNAL &&
      stream.ExternalWeights()) {
    utils::TwoFilesWriteStream &two_stream = dynamic_cast<utils::TwoFilesWriteStream &>(stream);
    int checked = 0;
    for (size_t i = 0; i < ref_external_data().size(); ++i) {
      const StringStringEntryProto &entry = ref_external_data()[i];
      if (entry.ref_key() == "location") {
        EXT_ENFORCE(!entry.ref_value().empty(), "External data location must not be empty.");
        checked += 1;
      } else if (entry.ref_key() == "size" || entry.ref_key() == "length") {
        int64_t size = entry.ref_value().toint64();
        EXT_ENFORCE(size == static_cast<int64_t>(ref_raw_data().size()), "Size mismatch ", size,
                    " != ", static_cast<int64_t>(ref_raw_data().size()), " name='",
                    ref_name().as_string(), "'");
        checked += 2;
      } else if (entry.ref_key() == "offset") {
        int64_t offset = entry.ref_value().toint64();
        EXT_ENFORCE(offset == two_stream.weights_size(), "Offset mismatch ", offset,
                    " != ", two_stream.weights_size(), " name ='", ref_name().as_string(), "'");
        checked += 4;
      }
    }
    EXT_ENFORCE(checked == 7,
                "External data is not fully specified. 'location', 'size', and 'offset' "
                "must be present in external_data, name='",
                ref_name().as_string(), "'");
    // TODO Checks sparse initializer as well.
  }
  WRITE_REPEATED_FIELD(options, stream, dims)
  WRITE_ENUM_FIELD(options, stream, data_type)
  WRITE_ENUM_FIELD(options, stream, data_location)
  WRITE_FIELD_NULL(options, stream, name)
  WRITE_FIELD_LIMIT(options, stream, raw_data)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, external_data)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, double_data)
  WRITE_REPEATED_FIELD(options, stream, float_data)
  WRITE_REPEATED_FIELD(options, stream, int32_data)
  WRITE_REPEATED_FIELD(options, stream, int64_data)
  WRITE_REPEATED_FIELD(options, stream, uint64_data)
  WRITE_REPEATED_FIELD(options, stream, string_data)
}
void TensorProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options) {
  READ_BEGIN(options, stream, TensorProto)                 //
  READ_REPEATED_FIELD(options, stream, dims)               //
  READ_ENUM_FIELD(options, stream, data_type)              //
  READ_OPTIONAL_ENUM_FIELD(options, stream, data_location) //
  READ_FIELD(options, stream, name)                        //
  READ_FIELD(options, stream, doc_string)                  //
  READ_FIELD_LIMIT_PARALLEL(options, stream, raw_data)     //
  READ_REPEATED_FIELD(options, stream, external_data)      //
  READ_REPEATED_FIELD(options, stream, metadata_props)     //
  READ_REPEATED_FIELD(options, stream, double_data)        //
  READ_REPEATED_FIELD(options, stream, float_data)         //
  READ_REPEATED_FIELD(options, stream, int32_data)         //
  READ_REPEATED_FIELD(options, stream, int64_data)         //
  READ_REPEATED_FIELD(options, stream, uint64_data)        //
  READ_REPEATED_FIELD(options, stream, string_data)        //
  READ_END(options, stream, TensorProto)                   //
                                         // After the reading, we need to check the data location.
  if (has_data_location() && ref_data_location() == DataLocation::EXTERNAL &&
      stream.ExternalWeights()) {
    utils::TwoFilesStream &two_stream = dynamic_cast<utils::TwoFilesStream &>(stream);
    offset_t offset = -1; // two_stream.second_tell();
    int64_t size = -1;

    for (size_t i = 0; i < ref_external_data().size(); ++i) {
      const StringStringEntryProto &entry = ref_external_data()[i];
      if (entry.ref_key() == "location") {
        EXT_ENFORCE(!entry.ref_value().empty(), "External data location must not be empty.");
        // Should check the value with the location of the second stream?
      } else if (entry.ref_key() == "length" || entry.ref_key() == "size") {
        size = entry.ref_value().toint64();
      } else if (entry.ref_key() == "offset") {
        offset = entry.ref_value().toint64();
      }
    }
    EXT_ENFORCE(offset >= 0 && size > 0, "External data offset and size must be specified, name='",
                ref_name().as_string(), "'");
    ref_raw_data().resize(size);
    if (options.parallel) {
      utils::DelayedBlock block;
      block.size = size;
      block.data = ref_raw_data().data();
      block.offset = two_stream.weights_tell();
      block.stream_id = 1; // The second stream is the weights stream.
      two_stream.ReadDelayedBlock(block);
    } else {
      two_stream.read_bytes_from_weights_stream(size, ref_raw_data().data(), offset);
    }
  }
}
std::vector<std::string> TensorProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(dims), NAME_EXIST_VALUE(data_type), NAME_EXIST_VALUE(data_location),
      NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(segment), NAME_EXIST_VALUE(raw_data),
      NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(external_data), NAME_EXIST_VALUE(metadata_props),
      NAME_EXIST_VALUE(double_data), NAME_EXIST_VALUE(float_data), NAME_EXIST_VALUE(int32_data),
      NAME_EXIST_VALUE(int64_data), NAME_EXIST_VALUE(uint64_data), NAME_EXIST_VALUE(string_data));
}

// SparseTensorProto

IMPLEMENT_PROTO(SparseTensorProto)
uint64_t SparseTensorProto::SerializeSize(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, values)
  SIZE_FIELD(size, options, stream, indices)
  SIZE_REPEATED_FIELD(size, options, stream, dims)
  return size;
}
void SparseTensorProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, values)
  WRITE_FIELD(options, stream, indices)
  WRITE_REPEATED_FIELD(options, stream, dims)
}
void SparseTensorProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, SparseTensorProto) //
    READ_FIELD(options, stream, values)            //
    READ_FIELD(options, stream, indices)           //
    READ_REPEATED_FIELD(options, stream, dims)     //
    READ_END(options, stream, SparseTensorProto)   //
} std::vector<std::string> SparseTensorProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(values), NAME_EXIST_VALUE(indices),
                                        NAME_EXIST_VALUE(dims));
}

// TypeProto::Tensor

IMPLEMENT_PROTO(TypeProto::Tensor)
uint64_t TypeProto::Tensor::SerializeSize(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, elem_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, shape)
  return size;
}
void TypeProto::Tensor::SerializeToStream(utils::BinaryWriteStream &stream,
                                          SerializeOptions &options) const {
  WRITE_FIELD(options, stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, shape)
}
void TypeProto::Tensor::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto::Tensor)    //
    READ_FIELD(options, stream, elem_type)            //
    READ_OPTIONAL_PROTO_FIELD(options, stream, shape) //
    READ_END(options, stream, TypeProto::Tensor)      //
} std::vector<std::string> TypeProto::Tensor::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::SparseTensor

IMPLEMENT_PROTO(TypeProto::SparseTensor)
uint64_t TypeProto::SparseTensor::SerializeSize(utils::BinaryWriteStream &stream,
                                                SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, elem_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, shape)
  return size;
}
void TypeProto::SparseTensor::SerializeToStream(utils::BinaryWriteStream &stream,
                                                SerializeOptions &options) const {
  WRITE_FIELD(options, stream, elem_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, shape)
}
void TypeProto::SparseTensor::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto::SparseTensor) //
    READ_FIELD(options, stream, elem_type)               //
    READ_OPTIONAL_PROTO_FIELD(options, stream, shape)    //
    READ_END(options, stream, TypeProto::SparseTensor)   //
} std::vector<std::string> TypeProto::SparseTensor::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(elem_type), NAME_EXIST_VALUE(shape));
}

// TypeProto::Sequence

IMPLEMENT_PROTO(TypeProto::Sequence)
uint64_t TypeProto::Sequence::SerializeSize(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, elem_type)
  return size;
}
void TypeProto::Sequence::SerializeToStream(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, elem_type)
}
void TypeProto::Sequence::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto::Sequence)      //
    READ_OPTIONAL_PROTO_FIELD(options, stream, elem_type) //
    READ_END(options, stream, TypeProto::Sequence)        //
} std::vector<std::string> TypeProto::Sequence::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(elem_type));
}

//  TypeProto::Map

IMPLEMENT_PROTO(TypeProto::Map)
uint64_t TypeProto::Map::SerializeSize(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, key_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, value_type)
  return size;
}
void TypeProto::Map::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, key_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, value_type)
}
void TypeProto::Map::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto::Map)            //
    READ_FIELD(options, stream, key_type)                  //
    READ_OPTIONAL_PROTO_FIELD(options, stream, value_type) //
    READ_END(options, stream, TypeProto::Map)              //
} std::vector<std::string> TypeProto::Map::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(key_type),
                                        NAME_EXIST_VALUE(value_type));
}

// TypeProto::Optional

IMPLEMENT_PROTO(TypeProto::Optional)
uint64_t TypeProto::Optional::SerializeSize(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, elem_type)
  return size;
}
void TypeProto::Optional::SerializeToStream(utils::BinaryWriteStream &stream,
                                            SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, elem_type)
}
void TypeProto::Optional::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto::Optional)      //
    READ_OPTIONAL_PROTO_FIELD(options, stream, elem_type) //
    READ_END(options, stream, TypeProto::Optional)        //
} std::vector<std::string> TypeProto::Optional::PrintToVectorString(utils::PrintOptions &options)
    const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(elem_type));
}

// TypeProto

IMPLEMENT_PROTO(TypeProto)
uint64_t TypeProto::SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, tensor_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sequence_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, map_type)
  SIZE_FIELD(size, options, stream, denotation)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sparse_tensor_type)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, optional_type)
  return size;
}
void TypeProto::SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sequence_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, map_type)
  WRITE_FIELD(options, stream, denotation)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_type)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, optional_type)
}
void TypeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, TypeProto)                         //
    READ_OPTIONAL_PROTO_FIELD(options, stream, tensor_type)        //
    READ_OPTIONAL_PROTO_FIELD(options, stream, sequence_type)      //
    READ_FIELD(options, stream, denotation)                        //
    READ_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor_type) //
    READ_OPTIONAL_PROTO_FIELD(options, stream, optional_type)      //
    READ_END(options, stream, TypeProto)                           //
} std::vector<std::string> TypeProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(tensor_type), NAME_EXIST_VALUE(sequence_type),
      NAME_EXIST_VALUE(map_type), NAME_EXIST_VALUE(denotation), NAME_EXIST_VALUE(sparse_tensor_type),
      NAME_EXIST_VALUE(optional_type));
}

// ValueInfoProto

IMPLEMENT_PROTO(ValueInfoProto)
uint64_t ValueInfoProto::SerializeSize(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, name)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, type)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void ValueInfoProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, type)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}

void ValueInfoProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, ValueInfoProto)          //
    READ_FIELD(options, stream, name)                    //
    READ_OPTIONAL_PROTO_FIELD(options, stream, type)     //
    READ_FIELD(options, stream, doc_string)              //
    READ_REPEATED_FIELD(options, stream, metadata_props) //
    READ_END(options, stream, ValueInfoProto)            //
} std::vector<std::string> ValueInfoProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(type),
                                        NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(metadata_props));
}

// AttributeProto

IMPLEMENT_PROTO(AttributeProto)
uint64_t AttributeProto::SerializeSize(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, ref_attr_name)
  SIZE_ENUM_FIELD(size, options, stream, type)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_FIELD(size, options, stream, f)
  SIZE_FIELD(size, options, stream, i)
  SIZE_FIELD_NULL(size, options, stream, s)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, t)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, sparse_tensor)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, g)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, tp)
  SIZE_REPEATED_FIELD(size, options, stream, floats)
  SIZE_REPEATED_FIELD(size, options, stream, ints)
  SIZE_REPEATED_FIELD(size, options, stream, strings)
  SIZE_REPEATED_FIELD(size, options, stream, tensors)
  SIZE_REPEATED_FIELD(size, options, stream, sparse_tensors)
  SIZE_REPEATED_FIELD(size, options, stream, graphs)
  return size;
}
void AttributeProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                       SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, ref_attr_name)
  WRITE_ENUM_FIELD(options, stream, type)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_FIELD(options, stream, f)
  WRITE_FIELD(options, stream, i)
  WRITE_FIELD_NULL(options, stream, s)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, t)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, g)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, tp)
  WRITE_REPEATED_FIELD(options, stream, floats)
  WRITE_REPEATED_FIELD(options, stream, ints)
  WRITE_REPEATED_FIELD(options, stream, strings)
  WRITE_REPEATED_FIELD(options, stream, tensors)
  WRITE_REPEATED_FIELD(options, stream, sparse_tensors)
  WRITE_REPEATED_FIELD(options, stream, graphs)
}
void AttributeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, AttributeProto)               //
    READ_FIELD(options, stream, name)                         //
    READ_FIELD(options, stream, ref_attr_name)                //
    READ_ENUM_FIELD(options, stream, type)                    //
    READ_FIELD(options, stream, doc_string)                   //
    READ_FIELD(options, stream, f)                            //
    READ_FIELD(options, stream, i)                            //
    READ_FIELD(options, stream, s)                            //
    READ_OPTIONAL_PROTO_FIELD(options, stream, t)             //
    READ_OPTIONAL_PROTO_FIELD(options, stream, sparse_tensor) //
    READ_OPTIONAL_PROTO_FIELD(options, stream, g)             //
    READ_OPTIONAL_PROTO_FIELD(options, stream, tp)            //
    READ_REPEATED_FIELD(options, stream, floats)              //
    READ_REPEATED_FIELD(options, stream, ints)                //
    READ_REPEATED_FIELD(options, stream, strings)             //
    READ_REPEATED_FIELD(options, stream, tensors)             //
    READ_REPEATED_FIELD(options, stream, sparse_tensors)      //
    READ_REPEATED_FIELD(options, stream, graphs)              //
    READ_END(options, stream, AttributeProto)                 //
} std::vector<std::string> AttributeProto::PrintToVectorString(utils::PrintOptions &options) const {
  switch (type_) {
  case AttributeType::UNDEFINED:
    return {MakeString("{", name_.as_string(), ": UNDEFINED }")};
  case AttributeType::FLOAT:
    return {MakeString("{", name_.as_string(), ": ", has_f() ? MakeString(*f_) : "?", "}")};
  case AttributeType::INT:
    return {MakeString("{", name_.as_string(), ": ", has_i() ? MakeString(*i_) : "?", "}")};
  case AttributeType::STRING:
    return {MakeString("{", name_.as_string(), ": ", s_.as_string(), "}")};
  case AttributeType::FLOATS:
    return {MakeString("{", name_.as_string(), ": ", write_as_string(options, floats_), "}")};
  case AttributeType::INTS:
    return {MakeString("{", name_.as_string(), ": ", write_as_string(options, ints_), "}")};
  case AttributeType::STRINGS:
    return {MakeString("{", name_.as_string(), ": ", write_as_string(options, strings_), "}")};
  default:
    return write_proto_into_vector_string(
        options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(ref_attr_name), NAME_EXIST_VALUE(doc_string),
        NAME_EXIST_VALUE(type), NAME_EXIST_VALUE(f), NAME_EXIST_VALUE(i), NAME_EXIST_VALUE(s),
        NAME_EXIST_VALUE(t), NAME_EXIST_VALUE(sparse_tensor), NAME_EXIST_VALUE(g),
        NAME_EXIST_VALUE(floats), NAME_EXIST_VALUE(ints), NAME_EXIST_VALUE(strings),
        NAME_EXIST_VALUE(tensors), NAME_EXIST_VALUE(sparse_tensors), NAME_EXIST_VALUE(graphs),
        NAME_EXIST_VALUE(tp));
  }
}

// NodeProto

IMPLEMENT_PROTO(NodeProto)
uint64_t NodeProto::SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_FIELD(size, options, stream, name)
  SIZE_FIELD(size, options, stream, op_type)
  SIZE_REPEATED_FIELD(size, options, stream, attribute)
  SIZE_FIELD_NULL(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, overload)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, device_configurations)
  return size;
}
void NodeProto::SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_FIELD(options, stream, name)
  WRITE_FIELD(options, stream, op_type)
  WRITE_REPEATED_FIELD(options, stream, attribute)
  WRITE_FIELD_NULL(options, stream, domain)
  WRITE_FIELD(options, stream, overload)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, device_configurations)
}
void NodeProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, NodeProto)                      //
    READ_REPEATED_FIELD(options, stream, input)                 //
    READ_REPEATED_FIELD(options, stream, output)                //
    READ_FIELD(options, stream, name)                           //
    READ_FIELD(options, stream, op_type)                        //
    READ_REPEATED_FIELD(options, stream, attribute)             //
    READ_FIELD(options, stream, domain)                         //
    READ_FIELD(options, stream, overload)                       //
    READ_FIELD(options, stream, doc_string)                     //
    READ_REPEATED_FIELD(options, stream, metadata_props)        //
    READ_REPEATED_FIELD(options, stream, device_configurations) //
    READ_END(options, stream, NodeProto)                        //
} std::vector<std::string> NodeProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(input), NAME_EXIST_VALUE(output), NAME_EXIST_VALUE(name),
      NAME_EXIST_VALUE(op_type), NAME_EXIST_VALUE(attribute), NAME_EXIST_VALUE(domain),
      NAME_EXIST_VALUE(overload), NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(metadata_props),
      NAME_EXIST_VALUE(device_configurations));
}

// GraphProto

IMPLEMENT_PROTO(GraphProto)
uint64_t GraphProto::SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_REPEATED_FIELD(size, options, stream, node)
  SIZE_FIELD(size, options, stream, name)
  SIZE_REPEATED_FIELD(size, options, stream, initializer)
  SIZE_REPEATED_FIELD(size, options, stream, sparse_initializer)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_REPEATED_FIELD(size, options, stream, value_info)
  SIZE_REPEATED_FIELD(size, options, stream, quantization_annotation)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void GraphProto::SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  WRITE_REPEATED_FIELD(options, stream, node)
  WRITE_FIELD(options, stream, name)
  WRITE_REPEATED_FIELD(options, stream, initializer)
  WRITE_REPEATED_FIELD(options, stream, sparse_initializer)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_REPEATED_FIELD(options, stream, value_info)
  WRITE_REPEATED_FIELD(options, stream, quantization_annotation)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}
void GraphProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, GraphProto)                       //
    READ_REPEATED_FIELD(options, stream, node)                    //
    READ_FIELD(options, stream, name)                             //
    READ_REPEATED_FIELD(options, stream, initializer)             //
    READ_REPEATED_FIELD(options, stream, sparse_initializer)      //
    READ_FIELD(options, stream, doc_string)                       //
    READ_REPEATED_FIELD(options, stream, input)                   //
    READ_REPEATED_FIELD(options, stream, output)                  //
    READ_REPEATED_FIELD(options, stream, value_info)              //
    READ_REPEATED_FIELD(options, stream, quantization_annotation) //
    READ_REPEATED_FIELD(options, stream, metadata_props)          //
    READ_END(options, stream, GraphProto)                         //  // NOLINT
} std::vector<std::string> GraphProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(input),
      NAME_EXIST_VALUE(output), NAME_EXIST_VALUE(metadata_props), NAME_EXIST_VALUE(node),
      NAME_EXIST_VALUE(initializer), NAME_EXIST_VALUE(sparse_initializer), NAME_EXIST_VALUE(value_info),
      NAME_EXIST_VALUE(quantization_annotation));
}

// FunctionProto

IMPLEMENT_PROTO(FunctionProto)
uint64_t FunctionProto::SerializeSize(utils::BinaryWriteStream &stream,
                                      SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, name)
  SIZE_REPEATED_FIELD(size, options, stream, input)
  SIZE_REPEATED_FIELD(size, options, stream, output)
  SIZE_REPEATED_FIELD(size, options, stream, attribute)
  SIZE_REPEATED_FIELD(size, options, stream, attribute_proto)
  SIZE_REPEATED_FIELD(size, options, stream, node)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_REPEATED_FIELD(size, options, stream, opset_import)
  SIZE_FIELD_NULL(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, overload)
  SIZE_REPEATED_FIELD(size, options, stream, value_info)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  return size;
}
void FunctionProto::SerializeToStream(utils::BinaryWriteStream &stream,
                                      SerializeOptions &options) const {
  WRITE_FIELD(options, stream, name)
  WRITE_REPEATED_FIELD(options, stream, input)
  WRITE_REPEATED_FIELD(options, stream, output)
  WRITE_REPEATED_FIELD(options, stream, attribute)
  WRITE_REPEATED_FIELD(options, stream, attribute_proto)
  WRITE_REPEATED_FIELD(options, stream, node)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_REPEATED_FIELD(options, stream, opset_import)
  WRITE_FIELD_NULL(options, stream, domain)
  WRITE_FIELD(options, stream, overload)
  WRITE_REPEATED_FIELD(options, stream, value_info)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
}
void FunctionProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, FunctionProto)            //
    READ_FIELD(options, stream, name)                     //
    READ_REPEATED_FIELD(options, stream, input)           //
    READ_REPEATED_FIELD(options, stream, output)          //
    READ_REPEATED_FIELD(options, stream, attribute)       //
    READ_REPEATED_FIELD(options, stream, attribute_proto) //
    READ_REPEATED_FIELD(options, stream, node)            //
    READ_FIELD(options, stream, doc_string)               //
    READ_REPEATED_FIELD(options, stream, opset_import)    //
    READ_FIELD(options, stream, domain)                   //
    READ_FIELD(options, stream, overload)                 //
    READ_REPEATED_FIELD(options, stream, value_info)      //
    READ_REPEATED_FIELD(options, stream, metadata_props)  //
    READ_END(options, stream, FunctionProto)              //  // NOLINT
} std::vector<std::string> FunctionProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(name), NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(overload),
      NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(input), NAME_EXIST_VALUE(output),
      NAME_EXIST_VALUE(opset_import), NAME_EXIST_VALUE(attribute), NAME_EXIST_VALUE(attribute_proto),
      NAME_EXIST_VALUE(node), NAME_EXIST_VALUE(value_info), NAME_EXIST_VALUE(metadata_props));
}

// ModelProto

IMPLEMENT_PROTO(ModelProto)
uint64_t ModelProto::SerializeSize(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  uint64_t size = 0;
  SIZE_FIELD(size, options, stream, ir_version)
  SIZE_REPEATED_FIELD(size, options, stream, opset_import)
  SIZE_FIELD(size, options, stream, producer_name)
  SIZE_FIELD(size, options, stream, producer_version)
  SIZE_FIELD(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, model_version)
  SIZE_FIELD(size, options, stream, doc_string)
  SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, graph)
  SIZE_REPEATED_FIELD(size, options, stream, metadata_props)
  SIZE_REPEATED_FIELD(size, options, stream, functions)
  SIZE_REPEATED_FIELD(size, options, stream, configuration)
  return size;
}
void ModelProto::SerializeToStream(utils::BinaryWriteStream &stream, SerializeOptions &options) const {
  WRITE_FIELD(options, stream, ir_version)
  WRITE_REPEATED_FIELD(options, stream, opset_import)
  WRITE_FIELD(options, stream, producer_name)
  WRITE_FIELD(options, stream, producer_version)
  WRITE_FIELD(options, stream, domain)
  WRITE_FIELD(options, stream, model_version)
  WRITE_FIELD(options, stream, doc_string)
  WRITE_OPTIONAL_PROTO_FIELD(options, stream, graph)
  WRITE_REPEATED_FIELD(options, stream, metadata_props)
  WRITE_REPEATED_FIELD(options, stream, functions)
  WRITE_REPEATED_FIELD(options, stream, configuration)
}
void ModelProto::ParseFromStream(utils::BinaryStream &stream, ParseOptions &options){
    READ_BEGIN(options, stream, ModelProto)              //
    READ_FIELD(options, stream, ir_version)              //
    READ_REPEATED_FIELD(options, stream, opset_import)   //
    READ_FIELD(options, stream, producer_name)           //
    READ_FIELD(options, stream, producer_version)        //
    READ_FIELD(options, stream, domain)                  //
    READ_FIELD(options, stream, model_version)           //
    READ_FIELD(options, stream, doc_string)              //
    READ_OPTIONAL_PROTO_FIELD(options, stream, graph)    //
    READ_REPEATED_FIELD(options, stream, metadata_props) //
    READ_REPEATED_FIELD(options, stream, functions)      //
    READ_REPEATED_FIELD(options, stream, configuration)  //
    READ_END(options, stream, ModelProto)                //  // NOLINT
} std::vector<std::string> ModelProto::PrintToVectorString(utils::PrintOptions &options) const {
  return write_proto_into_vector_string(
      options, NAME_EXIST_VALUE(ir_version), NAME_EXIST_VALUE(opset_import),
      NAME_EXIST_VALUE(producer_name), NAME_EXIST_VALUE(producer_version), NAME_EXIST_VALUE(domain),
      NAME_EXIST_VALUE(model_version), NAME_EXIST_VALUE(doc_string), NAME_EXIST_VALUE(graph),
      NAME_EXIST_VALUE(metadata_props), NAME_EXIST_VALUE(functions), NAME_EXIST_VALUE(configuration));
}

} // namespace onnx2
