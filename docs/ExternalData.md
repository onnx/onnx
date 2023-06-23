<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# External Data

## Loading an ONNX Model with External Data

* [Default] If the external data is under the same directory of the model, simply use `onnx.load()`

```python
import onnx

onnx_model = onnx.load("path/to/the/model.onnx")
```

* If the external data is under another directory, use `load_external_data_for_model()` to specify the directory path and load after using `onnx.load()`

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load("path/to/the/model.onnx", load_external_data=False)
load_external_data_for_model(onnx_model, "data/directory/path/")
# Then the onnx_model has loaded the external data from the specific directory
```

## Converting an ONNX Model to External Data
```python
import onnx
from onnx.external_data_helper import convert_model_to_external_data

onnx_model = ... # Your model in memory as ModelProto
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Must be followed by save_model to save the converted model to a specific path
onnx.save_model(onnx_model, "path/to/save/the/model.onnx")
# Then the onnx_model has converted raw data as external data and saved to specific directory
```

## Converting and Saving an ONNX Model to External Data

```python
import onnx

onnx_model = ... # Your model in memory as ModelProto
onnx.save_model(onnx_model, "path/to/save/the/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="filename", size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data and saved to specific directory
```

## onnx.checker for Models with External Data

### Models with External Data (<2GB)

Current checker supports checking models with external data. Specify either loaded onnx model or model path to the checker.

### Large models >2GB

However, for those models larger than 2GB, please use the model path for onnx.checker and the external data needs to be under the same directory.

```python
import onnx

onnx.checker.check_model("path/to/the/model.onnx")
# onnx.checker.check_model(loaded_onnx_model) will fail if given >2GB model
```

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
