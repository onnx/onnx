# External Data

## Loading an ONNX Model with External Data
```python
import onnx

onnx_model = onnx.load('path/to/the/model.onnx')
# `onnx_model` is a ModelProto struct
```
Runnable IPython notebooks:
- [load_model.ipynb](https://github.com/onnx/onnx/tree/master/onnx/examples/load_model.ipynb)

## TensorProto: data_location and external_data fields

There are two fields related to the external data in TensorProto message type.

### data_location field
`data_location` field stores the location of data for this tensor. Value MUST be one of:
* `MESSAGE` - data stored in type-specific fields inside the protobuf message.
* `RAW` - data stored in raw_data field.
* `EXTERNAL` - data stored in an external location as described by external_data field.
* `value` not set - legacy value. Assume data is stored in raw_data (if set) otherwise in message.

### external_data field
`external_data` field stores key-value pairs of strings describing data location

Recognized keys are:

* `"location"` (required) - file path relative to the filesystem directory where the ONNX protobuf model was stored. Up-directory path components such as .. are disallowed and should be stripped when parsing.
* `"offset"` (optional) - position of byte at which stored data begins. Integer stored as string. Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
* `"length"` (optional) - number of bytes containing data. Integer stored as string.
* `"checksum"` (optional) - SHA1 digest of file specified in under 'location' key.

After an ONNX file is loaded, all `external_data` fields may be updated with an additional key `("basepath")`, which stores the path to the directory from which he ONNX model file was loaded.

### External data files
Data stored in external data files will be in the same binary bytes string format as is used by the `raw_data` field in current ONNX implementations.

Reference
https://github.com/onnx/onnx/pull/678
