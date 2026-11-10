# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

target_sources(onnx PRIVATE
    attr_proto_util.cc
    attr_proto_util.h
    data_propagators.h
    data_type_utils.cc
    data_type_utils.h
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

# Operator schemas generated from the spec files in operator-specifications.
# One translation unit is generated per (domain, operator).
set(SCHEMA_SPECS_DIR "${ONNX_ROOT}/operator-specifications")
set(GEN_SCHEMAS_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(OPERATOR_SPECS_TOOL_DIR "${ONNX_ROOT}/tools/operator_specs")
file(GLOB_RECURSE SCHEMA_SPEC_FILES CONFIGURE_DEPENDS "${SCHEMA_SPECS_DIR}/*.md")
# Only <domain>/v<N>/<op>.md are spec files (e.g., not README.md).
list(FILTER SCHEMA_SPEC_FILES INCLUDE REGEX "/[^/]+/v[0-9]+/[^/]+\\.md$")
# Regenerate if spec files are removed; this file only changes with the list.
string(JOIN "\n" SCHEMA_SPEC_FILES_CONTENT ${SCHEMA_SPEC_FILES})
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema_spec_files.txt" CONTENT "${SCHEMA_SPEC_FILES_CONTENT}")
file(GLOB_RECURSE OPERATOR_SPECS_TOOL_FILES CONFIGURE_DEPENDS
     "${OPERATOR_SPECS_TOOL_DIR}/*.py" "${OPERATOR_SPECS_TOOL_DIR}/*.jinja")
set(GEN_SCHEMAS_SOURCES)
foreach(spec_file IN LISTS SCHEMA_SPEC_FILES)
  # <domain>/v<N>/<op>.md -> <domain with '_' for '.'>/<op>.cc
  file(RELATIVE_PATH rel "${SCHEMA_SPECS_DIR}" "${spec_file}")
  string(REGEX REPLACE "^([^/]+)/v[0-9]+/([^/]+)\\.md$" "\\1/\\2.cc" rel "${rel}")
  string(REPLACE "." "_" rel_dir "${rel}")
  string(REGEX REPLACE "_cc$" ".cc" rel "${rel_dir}")
  list(APPEND GEN_SCHEMAS_SOURCES "${GEN_SCHEMAS_DIR}/${rel}")
endforeach()
list(REMOVE_DUPLICATES GEN_SCHEMAS_SOURCES)
add_custom_command(
  OUTPUT ${GEN_SCHEMAS_SOURCES} "${GEN_SCHEMAS_DIR}/schemas.h"
  COMMAND ${ONNX_PYTHON_INTERPRETER} -m operator_specs
          --input "${SCHEMA_SPECS_DIR}" --output "${GEN_SCHEMAS_DIR}"
  WORKING_DIRECTORY "${ONNX_ROOT}/tools"
  DEPENDS ${SCHEMA_SPEC_FILES} "${CMAKE_CURRENT_BINARY_DIR}/schema_spec_files.txt" ${OPERATOR_SPECS_TOOL_FILES}
  COMMENT "Generating operator schemas from spec files"
  VERBATIM)
# The custom command is defined in a different directory than the 'onnx'
# target; a custom target in this directory owns it.
add_custom_target(onnx_gen_schemas DEPENDS ${GEN_SCHEMAS_SOURCES} "${GEN_SCHEMAS_DIR}/schemas.h")
add_dependencies(onnx onnx_gen_schemas)
target_sources(onnx PRIVATE ${GEN_SCHEMAS_SOURCES} "${GEN_SCHEMAS_DIR}/schemas.h")

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
    TARGET_DIRECTORY onnx
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
