<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX File Format Proposal

## Summary

We propose a new file format for ONNX models that is a specific application of the [zip](https://en.wikipedia.org/wiki/Zip_(file_format)) file format. We would like to address issues with capacity limits as well as (de)serialization inefficiencies[0][1]. We aim to design a file format that is simple, widely applicable, and efficient. By storing Tensor values (i.e. values typically contained in `TensorProto` messages) as files within a zip archive, we avoid these size limitations and—with special constraints—allow for direct memory-mapping of an ONNX file such that weights can be used directly from the memory-mapped region. Using zip as our base file format allows us to create a design that is conceptually simple as well as well-supported on various platforms.

## Design

We propose to treat a .zip file as a key-value store, mapping string keys (filenames) to binary data files. For ONNX model serialization, we will have the following entries:


* Data files - Files mapping a unique string identifier to a raw binary data file. These files shall be referenced from the appropriate fields within the base `ModelProto`
* `__MODEL_PROTO` - File that contains the `ModelProto` describing the file


Note that the order is significant here. We place the model definition file at the end of the archive to allow for the common case of net manipulations while keeping the weights invariant. This way, tools that manipulate the archive do not need to repack or realign all weights when only touching the model file.

Within the ONNX protobuf definition, we propose the following changes:


* Add `optional string external_data` to `TensorProto`. This can be treated as a data field similar to `float_data`, `int_data`, etc in that there must be exactly one of those fields specified. If a `TensorProto` specifies `external_data`, the implementation shall resolve this reference by string key in the containing zip archive. All values of `external_data` must be unique (under down-casing) and conform to the C identifier specification.


Raw data files referenced by `TensorProto`s shall conform to the following specification:


* The data shall be equivalent to that stored within the `raw_data` field in `TensorProto`.
* Raw data files within the zip archive shall reside on an alignment boundary of 64 bytes. That is, the byte offset within the file of the first byte of a raw data tensor must be divisible by 64. This requirement can be fulfilled by packing bytes into the `extra` field of each local file record in the zip archive. (example: [2]). This constraint facilitates the direct memory-mapping of data files within the archive, and allows for architectures with both strict alignment requirements (e.g. SIMD instructions on aligned data) to operate and give architectures that operate more efficiently on cache line-aligned data to take full advantage.

## File Extension

In keeping with other domain-specific zip applications, we propose to use a custom file extension rather than the `.zip` extension. A custom file extension makes it clear to the user that this is not a general zip file, but rather a file that should be emitted by ONNX tools to conform to the spec.


## Future-Proofing Considerations

This file format represents a generic key-value store that is scalable to many entries as well as large values. Further improvements to the format may come in the form of supporting different or multiple model definitions within the same model, or modifying the way in which weight files are stored. Building off of a proven archival format allows us the reliability as well as flexibility of zip.



[0] https://github.com/onnx/onnx/issues/251
[1] https://stackoverflow.com/questions/34128872/google-protobuf-maximum-size
[2] https://developer.android.com/studio/command-line/zipalign.html implementation https://github.com/aosp-mirror/platform_build/blob/master/tools/zipalign/ZipAlign.cpp
