# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

target_sources(onnx_core PRIVATE
    attr_proto_util.cc
    attr_proto_util.h
    data_propagators.h
    data_type_utils.cc
    data_type_utils.h
    doc_strings.cc
    doc_strings.h
    function.cc
    function.h
    operator_sets.h
    operator_sets_ml.h
    operator_sets_preview.h
    operator_sets_training.h
    parser.cc
    parser.h
    printer.cc
    printer.h
    schema.cc
    schema.h
    shape_inference.cc
    shape_inference.h
    tensor_proto_util.cc
    tensor_proto_util.h
    tensor_util.cc
    tensor_util.h
)

add_subdirectory(controlflow)
add_subdirectory(generator)
add_subdirectory(image)
add_subdirectory(logical)
add_subdirectory(math)
add_subdirectory(nn)
add_subdirectory(object_detection)
add_subdirectory(optional)
add_subdirectory(preview)
add_subdirectory(quantization)
add_subdirectory(reduction)
add_subdirectory(rnn)
add_subdirectory(sequence)
add_subdirectory(tensor)
add_subdirectory(text)
add_subdirectory(traditionalml)
add_subdirectory(training)

# These files exhibit issues in unity builds, so we disable them.
set_source_files_properties(schema.cc tensor_proto_util.cc
    TARGET_DIRECTORY onnx_core
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
