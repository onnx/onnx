/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/shape_inference/implementation.h"
#include <fstream>
#include <list>
#include "onnx/checker.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {
namespace {

std::string getValueCaseString(const TypeProto& type) {
  switch (type.value_case()) {
    case TypeProto::ValueCase::kTensorType:
      return "tensor_type";
    case TypeProto::ValueCase::kSequenceType:
      return "sequence_type";
    case TypeProto::ValueCase::kMapType:
      return "map_type";
    case TypeProto::ValueCase::kOptionalType:
      return "optional_type";
#ifdef ONNX_ML
    case TypeProto::ValueCase::kOpaqueType:
      return "opaque_type";
#endif
    case TypeProto::ValueCase::kSparseTensorType:
      return "sparse_tensor_type";
    case TypeProto::ValueCase::VALUE_NOT_SET:
      return "NOT_SET";
    default:
      return ONNX_NAMESPACE::to_string(type.value_case());
  }
}

std::string getElemTypeString(const TypeProto_Tensor& type) {
#ifndef ONNX_USE_LITE_PROTO
  const std::string type_str = TensorProto::DataType_Name(static_cast<TensorProto_DataType>(type.elem_type()));
  if (!type_str.empty()) {
    return type_str;
  }
#endif
  return ONNX_NAMESPACE::to_string(type.elem_type());
}

std::string getElemTypeString(const TypeProto_SparseTensor& type) {
#ifndef ONNX_USE_LITE_PROTO
  const std::string type_str = TensorProto::DataType_Name(static_cast<TensorProto_DataType>(type.elem_type()));
  if (!type_str.empty()) {
    return type_str;
  }
#endif
  return ONNX_NAMESPACE::to_string(type.elem_type());
}

} // namespace

template<class TensorTypeProto>
void checkTensorShapesAndTypes(const TensorTypeProto& inferredType, const TensorTypeProto& existingType) {
  if (inferredType.elem_type() != TensorProto::UNDEFINED && existingType.elem_type() != TensorProto::UNDEFINED &&
      existingType.elem_type() != inferredType.elem_type()) {
    std::stringstream ss;
    ss << "Inferred elem type differs from existing elem type: (" << getElemTypeString(inferredType) << ") vs ("
       << getElemTypeString(existingType) << ")";
    fail_type_inference(ss.str());
  }

  if (!inferredType.has_shape() || !existingType.has_shape()) {
    return;
  }

  if (inferredType.shape().dim_size() != existingType.shape().dim_size()) {
    std::stringstream ss;
    ss << "Inferred shape and existing shape differ in rank: (" << inferredType.shape().dim_size() << ") vs ("
       << existingType.shape().dim_size() << ")";
    fail_shape_inference(ss.str());
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    const auto& existingDim = existingType.shape().dim(i);
    if (inferredDim.has_dim_value() && existingDim.has_dim_value() &&
        inferredDim.dim_value() != existingDim.dim_value()) {
      std::stringstream ss;
      ss << "Inferred shape and existing shape differ in dimension " << i << ": (" << inferredDim.dim_value()
         << ") vs (" << existingDim.dim_value() << ")";
      fail_shape_inference(ss.str());
    }
  }
}

void checkShapesAndTypes(const TypeProto& inferredType, const TypeProto& existingType) {
  const auto inferredTypeCase = inferredType.value_case();
  const auto existingTypeCase = existingType.value_case();
  if (inferredTypeCase == TypeProto::ValueCase::VALUE_NOT_SET ||
      existingTypeCase == TypeProto::ValueCase::VALUE_NOT_SET) {
    // nothing to check; will assign inferredType to undefined existingType
    return;
  }
  if (inferredTypeCase != existingTypeCase) {
    fail_type_inference(
        "type case mismatch. existing=",
        getValueCaseString(existingType),
        " inferred=",
        getValueCaseString(inferredType));
  }

  if (inferredTypeCase == TypeProto::kTensorType && existingTypeCase == TypeProto::kTensorType) {
    checkTensorShapesAndTypes(inferredType.tensor_type(), existingType.tensor_type());
  } else if (inferredTypeCase == TypeProto::kSparseTensorType && existingTypeCase == TypeProto::kSparseTensorType) {
    checkTensorShapesAndTypes(inferredType.sparse_tensor_type(), existingType.sparse_tensor_type());
  } else if (inferredTypeCase == TypeProto::kSequenceType && existingTypeCase == TypeProto::kSequenceType) {
    checkShapesAndTypes(inferredType.sequence_type().elem_type(), existingType.sequence_type().elem_type());
  } else if (inferredTypeCase == TypeProto::kOptionalType && existingTypeCase == TypeProto::kOptionalType) {
    checkShapesAndTypes(inferredType.optional_type().elem_type(), existingType.optional_type().elem_type());
  } else {
    fail_type_inference("type case unsupported. existing=", existingTypeCase, " inferred=", inferredTypeCase);
  }
}

// TypeProto_Tensor or TypeProto_SparseTensor
template <typename TensorTypeProto>
void generateSymbolicShape(TensorTypeProto* inferredType, SymbolTable& symbolTable) {
  if (!inferredType->has_shape()) {
    return;
  }
  for (int i = 0; i < inferredType->shape().dim_size(); ++i) {
    // set a symbol if it doesn't have dim_value and dim_param
    auto* dim = inferredType->mutable_shape()->mutable_dim(i);
    if (!dim->has_dim_value() && !dim->has_dim_param()) {
      dim->set_dim_param(symbolTable.createNew("unk__"));
    }
  }
}

void materializeSymbolicShape(TypeProto* inferredType, SymbolTable& symbolTable) {
  const auto inferred_val_case = inferredType->value_case();
  if (inferred_val_case == TypeProto::kTensorType) {
    generateSymbolicShape(inferredType->mutable_tensor_type(), symbolTable);
  } else if (inferred_val_case == TypeProto::kSparseTensorType) {
    generateSymbolicShape(inferredType->mutable_sparse_tensor_type(), symbolTable);
  } else if (inferred_val_case == TypeProto::kSequenceType) {
    materializeSymbolicShape(inferredType->mutable_sequence_type()->mutable_elem_type(), symbolTable);
  } else if (inferred_val_case == TypeProto::kOptionalType) {
    materializeSymbolicShape(inferredType->mutable_optional_type()->mutable_elem_type(), symbolTable);
  } else {
    fail_shape_inference("type case unsupported for symbolic shape inference. inferred=", inferred_val_case);
  }
}

void mergeShapesAndTypes(const TypeProto_Tensor& inferredType, TypeProto_Tensor* existingType) {
  if (existingType->elem_type() == TensorProto::UNDEFINED) {
    existingType->set_elem_type(inferredType.elem_type());
  }

  if (!inferredType.has_shape()) {
    return;
  }

  if (!existingType->has_shape()) {
    // Ensure the shape is initialized. Note that this must be done
    // even for (zero-dimensional) scalars.
    existingType->mutable_shape();

    for (int j = 0; j < inferredType.shape().dim_size(); ++j) {
      existingType->mutable_shape()->add_dim();
    }
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    auto* existingDim = existingType->mutable_shape()->mutable_dim(i);
    if (!existingDim->has_dim_value()) {
      *existingDim = inferredDim;
    }
  }
}

void mergeShapesAndTypes(const TypeProto_SparseTensor& inferredType, TypeProto_SparseTensor* existingType) {
  if (existingType->elem_type() == TensorProto::UNDEFINED) {
    existingType->set_elem_type(inferredType.elem_type());
  }

  if (!inferredType.has_shape()) {
    return;
  }

  if (!existingType->has_shape()) {
    // Ensure the shape is initialized. Note that this must be done
    // even for (zero-dimensional) scalars.
    existingType->mutable_shape();

    for (int j = 0; j < inferredType.shape().dim_size(); ++j) {
      existingType->mutable_shape()->add_dim();
    }
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    auto* existingDim = existingType->mutable_shape()->mutable_dim(i);
    if (!existingDim->has_dim_value()) {
      *existingDim = inferredDim;
    }
  }
}

void mergeShapesAndTypes(const TypeProto& inferredType, TypeProto* existingType) {
  // Check before merge
  checkShapesAndTypes(inferredType, *existingType);
  const auto inferred_val_case = inferredType.value_case();
  if (inferred_val_case == TypeProto::kTensorType) {
    mergeShapesAndTypes(inferredType.tensor_type(), existingType->mutable_tensor_type());
  } else if (inferred_val_case == TypeProto::kSparseTensorType) {
    mergeShapesAndTypes(inferredType.sparse_tensor_type(), existingType->mutable_sparse_tensor_type());
  } else if (inferred_val_case == TypeProto::kSequenceType) {
    mergeShapesAndTypes(
        inferredType.sequence_type().elem_type(), existingType->mutable_sequence_type()->mutable_elem_type());
  } else if (inferred_val_case == TypeProto::kOptionalType) {
    mergeShapesAndTypes(
        inferredType.optional_type().elem_type(), existingType->mutable_optional_type()->mutable_elem_type());
  }
}

static void InferShapesImpl(
    GraphProto* g,
    const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name,
    const std::unordered_map<std::string, int>& opset_imports,
    const ShapeInferenceOptions& options,
    SymbolTable* symbolTable,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const int ir_version = IR_VERSION // default the latest one
) {
  std::unordered_map<std::string, TypeProto*> valueTypesByName{outer_scope_value_types_by_name};
  std::unordered_map<std::string, TypeProto*> undefinedValueTypesByName{outer_scope_value_types_by_name};
  std::unordered_map<std::string, TensorShapeProto> generatedShapeDataByName;

  GraphInferenceContext graphInferenceContext{valueTypesByName, opset_imports, symbolTable, schema_registry, ir_version};
  for (auto& vi : *g->mutable_value_info()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *g->mutable_input()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *g->mutable_output()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    } else {
      // Some output type might be undefined in subgraph. e.g., Loop Op
      // Saving names of outputs with undefined types to allow assigning inferred types to them
      undefinedValueTypesByName[vi.name()] = vi.mutable_type();
    }
  }

  // save for free memory and preserve the address of the dynamically allocated
  // TypeProto
  std::list<TypeProto> initializerTypeList;

  std::unordered_map<std::string, const TensorProto*> inputDataByName;
  for (const auto& tp : g->initializer()) {
    inputDataByName[tp.name()] = &tp;
    // Consider the tensors from the initializer
    TypeProto initializerType;
    TypeProto_Tensor* initializerTensorType = initializerType.mutable_tensor_type();
    initializerTensorType->set_elem_type(tp.data_type());
    // set the shape according to the initializer shape info
    TensorShapeProto* shape = initializerTensorType->mutable_shape();
    for (int i = 0; i < tp.dims_size(); ++i) {
      shape->add_dim()->set_dim_value(tp.dims(i));
    }

    auto iter = valueTypesByName.find(tp.name());
    // If it already exists in input, check input and initializer is sync
    // use shape info from input (input has priority over initializer)
    if (iter != valueTypesByName.end()) {
      checkTensorShapesAndTypes(*initializerTensorType, *iter->second->mutable_tensor_type());
    }
    // Support IR>=4: some tensors can only exist in initializer and not in input
    // So shape_inference should make use of initializer shapes
    // Store initializer shape info in value_info as well
    else if (ir_version >= 4) {
      initializerTypeList.push_back(std::move(initializerType));
      valueTypesByName[tp.name()] = &initializerTypeList.back();
    }
  }

  std::unordered_map<std::string, const SparseTensorProto*> inputSparseDataByName;
  for (const auto& tp : g->sparse_initializer()) {
    const auto& name = tp.values().name();
    inputSparseDataByName[name] = &tp;
    // Consider the tensors from the initializer
    TypeProto initializerType;
    TypeProto_SparseTensor* initializerSparseTensorType = initializerType.mutable_sparse_tensor_type();
    initializerSparseTensorType->set_elem_type(tp.values().data_type());
    // set the shape according to the initializer shape info
    TensorShapeProto* shape = initializerSparseTensorType->mutable_shape();
    for (int i = 0; i < tp.dims_size(); ++i) {
      shape->add_dim()->set_dim_value(tp.dims(i));
    }

    auto iter = valueTypesByName.find(name);
    // If it already exists in input, check input and initializer is sync
    // use shape info from input (input has priority over initializer)
    if (iter != valueTypesByName.end()) {
      checkTensorShapesAndTypes(*initializerSparseTensorType, *iter->second->mutable_sparse_tensor_type());
    }
    // Support IR>=4: some tensors can only exist in initializer and not in input
    // So shape_inference should make use of initializer shapes
    // Store initializer shape info in value_info as well
    else if (ir_version >= 4) {
      initializerTypeList.push_back(std::move(initializerType));
      valueTypesByName[name] = &initializerTypeList.back();
    }
  }

  bool has_experimental_op = false;
  // If encounter experimental op, stop checking
  for (const auto& n : g->node()) {
    if (checker::check_is_experimental_op(n.op_type())) {
      std::cerr << "Warning: Shape inference does not support"
                << " models with experimental operators: " << n.op_type() << std::endl;
      has_experimental_op = true;
    }
  }

  // Collect data from constant nodes.
  for (const auto& n : g->node()) {
    if (n.op_type() != "Constant" || n.output().size() != 1) {
      continue;
    }
    for (const auto& attr : n.attribute()) {
      if (attr.name() == "value") {
        if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
          inputDataByName[n.output(0)] = &attr.t();
        } else if (attr.type() == AttributeProto::SPARSE_TENSOR && attr.has_sparse_tensor()) {
          inputSparseDataByName[n.output(0)] = &attr.sparse_tensor();
        }
      }
    }
  }

  std::vector<std::string> inference_errors;
  bool has_unsupported_op = false; // check whether exist unsupported ops

  for (auto& n : *g->mutable_node()) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      continue;
    }
    auto domain_version = dit->second;
    const auto schema = schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    InferenceContextImpl ctx(n, valueTypesByName, inputDataByName,inputSparseDataByName,
      &generatedShapeDataByName, &graphInferenceContext);
    if (!schema) {
      std::cerr << "Warning: Unsupported operator " << n.op_type() << ". No schema registered for this operator."
                << std::endl;
      has_unsupported_op = true;
      continue;
    } else if (schema->has_type_and_shape_inference_function()) {
      ONNX_TRY {
        schema->GetTypeAndShapeInferenceFunction()(ctx);
      }
      ONNX_CATCH(const ONNX_NAMESPACE::InferenceError& ex) {
        ONNX_HANDLE_EXCEPTION([&]() {
          // checker does not support unsupported/experimental operators
          // so it won't consider it as an error
          if (!has_unsupported_op && !has_experimental_op) {
            inference_errors.push_back(getErrorWithNodeInfo(n, ex));
          }
        });
        // Continue with inference for remaining nodes
        continue;
      }
    } else if (schema->HasFunction()) {
      ONNX_TRY {
        const auto func_proto = schema->GetFunction();
        std::unordered_map<std::string, int> function_opset_imports;
        for (const auto& opset_import : func_proto->opset_import()) {
            function_opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
        }

        InferShapeForFunctionNode(
            func_proto, function_opset_imports, schema_registry, ctx, options, symbolTable, &generatedShapeDataByName);
      }
      ONNX_CATCH(const ONNX_NAMESPACE::InferenceError& ex) {
        ONNX_HANDLE_EXCEPTION([&]() {
          // checker does not support unsupported/experimental operators
          // so it won't consider it as an error
          if (!has_unsupported_op && !has_experimental_op) {
            inference_errors.push_back(getErrorWithNodeInfo(n, ex));
          }
        });
        // Continue with inference for remaining nodes
        continue;
      }
    } else {
      // Continue with inference for remaining nodes
      continue;
    }

    ONNX_TRY {
      // check the type-equality for input and output
      if (options.check_type) {
        schema->CheckInputOutputType(ctx);
      }
      for (int i = 0; i < n.output_size(); ++i) {
        auto* inferredType = ctx.getOutputType(i);
        if (inferredType->value_case() == TypeProto::ValueCase::VALUE_NOT_SET) {
          continue;
        }

        // Find any pre-existing type and shape info. If there is such,
        // then check for compatibility with the inferred
        // information. Otherwise, initialize it in an empty state.
        auto iter = valueTypesByName.find(n.output(i));
        TypeProto* existingType = nullptr;
        if (iter != valueTypesByName.end()) {
          existingType = iter->second;
        } else {
          // Create a new value_info if defined type does not exist
          auto vi = g->add_value_info();
          vi->set_name(n.output(i));
          existingType = vi->mutable_type();
          // For undefined output type, update both value_info and output for now
          // Update existing output with undefined type: assign inferred type to it
          iter = undefinedValueTypesByName.find(n.output(i));
          if (iter != undefinedValueTypesByName.end()) {
            *iter->second = *inferredType;
          }
        }

        if (symbolTable) {
          materializeSymbolicShape(inferredType, *symbolTable);
        }

        // Now we can merge pre-existing and inferred info
        mergeShapesAndTypes(*inferredType, existingType);
        if (options.enable_data_propagation && schema->has_data_propagation_function()) {
          DataPropagationContextImpl dataPropagationCtx(
              n, valueTypesByName, inputDataByName, generatedShapeDataByName);
          schema->GetDataPropagationFunction()(dataPropagationCtx);
        }
        // Make merged info available to further inference.
        valueTypesByName[n.output(i)] = existingType;
      }
    }
    ONNX_CATCH(const std::runtime_error& err) {
      ONNX_HANDLE_EXCEPTION([&]() {
        fail_shape_inference(getErrorWithNodeInfo(n, err));
      });
    }
  }
  // Throw shape inference error if any
  // Error node right now only supports 0 and 1
  // When set to 0, any node level shape inference errors
  // are not thrown. This is to support backward compatiblity
  // with 1.7 and earlier releases. When set to 1 it will throw
  // all exceptions.
  // TODO: Add a more granular way for exception handling.
  if (options.error_mode > 0 && !inference_errors.empty()) {
    std::string full_errors = "Shape inference error(s): ";
    for (const std::string& error : inference_errors) {
      full_errors += error + "\n";
    }
    fail_shape_inference(full_errors);
  }
}

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options) {
  SymbolTableImpl symbolTable;
  traverseGraphsToAddExistingSymbols(*g, symbolTable);
  InferShapesImpl(
      g, std::unordered_map<std::string, TypeProto*>(0), opset_imports, options, &symbolTable, schema_registry);
}

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options) {
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : m.opset_import()) {
    opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
  }
  auto* g = m.mutable_graph();
  SymbolTableImpl symbolTable;
  traverseGraphsToAddExistingSymbols(*g, symbolTable);
  InferShapesImpl(
      g,
      std::unordered_map<std::string, TypeProto*>(0),
      opset_imports,
      options,
      &symbolTable,
      schema_registry,
      m.ir_version());
}

void InferShapes(
    const std::string& model_path,
    const std::string& save_path,
    const ISchemaRegistry* schema_registry,
    const ShapeInferenceOptions& options) {
  ModelProto model;
  std::fstream model_stream(model_path, std::ios::in | std::ios::binary);
  if (!model_stream.good()) {
    fail_check("Unable to open model file:", model_path, ". Please check if it is a valid file.");
  }
  std::string data{std::istreambuf_iterator<char>{model_stream}, std::istreambuf_iterator<char>{}};
  if (!ParseProtoFromBytes(&model, data.c_str(), data.size())) {
    fail_check(
        "Unable to parse model from file:", model_path, ". Please check if it is a valid protobuf file of model.");
  }
  InferShapes(model, schema_registry, options);
  // Save the inferred model to the original model path
  // Use SerializeToString instead of SerializeToOstream due to LITE_PROTO
  std::fstream output(save_path, std::ios::out | std::ios::trunc | std::ios::binary);
  std::string model_string;
  ONNX_TRY {
    model.SerializeToString(&model_string);
    output << model_string;
  }
  ONNX_CATCH(...) {
    fail_check("Unable to save inferred model to the target path:", save_path);
  }
}

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const std::unordered_map<std::string, int>& func_opset_imports,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options,
    SymbolTable* symbolTable,
    std::unordered_map<std::string, TensorShapeProto>* generatedShapeDataByName) {
  if (options.enable_data_propagation && generatedShapeDataByName == nullptr) {
    fail_shape_inference("Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
  }

  GraphProto g;
  // Get a temporary tensor-shape map
  const auto num_func_inputs = func->input_size();
  std::unordered_map<std::string, TypeProto*> temp_valueTypesByName;
  std::vector<TypeProto> temp_types_cache(func->input_size());
  for (int i = 0; i < num_func_inputs; ++i) {
    temp_types_cache[i] = *ctx.getInputType(i);
    temp_valueTypesByName[func->input().Get(i)] = &temp_types_cache[i];
  }
  // Get a temporary initial value map
  std::unordered_map<std::string, const TensorProto*> temp_initializersByName;
  std::unordered_map<std::string, const SparseTensorProto*> temp_SparseInitializersByName;
  for (int i = 0; i < static_cast<int>(ctx.getNumInputs()) && i < num_func_inputs; ++i) {
    const TypeProto* type = ctx.getInputType(i);
    if (type->value_case() == TypeProto::kTensorType && ctx.getInputData(i) != nullptr) {
      temp_initializersByName[func->input().Get(i)] = ctx.getInputData(i);
    } else if (type->value_case() == TypeProto::kSparseTensorType && 
               ctx.getInputSparseData(i) != nullptr) {
      temp_SparseInitializersByName[func->input().Get(i)] = ctx.getInputSparseData(i);
    }
  }
  std::unordered_map<std::string, const AttributeProto*> attr_map;
  for (auto& attr : func->attribute()) {
    if (ctx.getAttribute(attr) != nullptr) {
      attr_map[attr] = ctx.getAttribute(attr);
    }
  }

  for (auto& n : func->node()) {
    // Resolve domain for node
    auto it = func_opset_imports.find(n.domain());
    if (it == func_opset_imports.end()) {
      return;
    }
    auto domain_version = it->second;
    const auto schema = schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    if (!schema) {
      return;
    }
    NodeProto copy_n(n);
    // Add attribute information into the temporary node
    copy_n.clear_attribute();
    for (const auto& attr : n.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          auto copy_attr = *attr_map[attr.ref_attr_name()];
          copy_attr.set_name(attr.name());
          copy_n.add_attribute()->CopyFrom(copy_attr);
        }
      } else {
        copy_n.add_attribute()->CopyFrom(attr);
      }
    }

    InferenceContextImpl temp_ctx(copy_n, temp_valueTypesByName, temp_initializersByName,
      temp_SparseInitializersByName, generatedShapeDataByName);
    schema->GetTypeAndShapeInferenceFunction()(temp_ctx);
    for (int i = 0; i < copy_n.output_size(); ++i) {
      TypeProto* inferred_output_type = temp_ctx.getOutputType(i);
      const auto type_value_case = inferred_output_type->value_case();
      if (type_value_case == TypeProto::kTensorType) {
        const auto& tensor_type = inferred_output_type->tensor_type();
        // Bail out early if shape inference does nothing useful.
        if (tensor_type.elem_type() == TensorProto::UNDEFINED && !tensor_type.has_shape()) {
          continue;
        }
      } else if (type_value_case == TypeProto::kSparseTensorType) {
        const auto& sparse_tensor_type = inferred_output_type->sparse_tensor_type();
        // Bail out early if shape inference does nothing useful.
        if (sparse_tensor_type.elem_type() == TensorProto::UNDEFINED && !sparse_tensor_type.has_shape()) {
          continue;
        }
      } else {
        continue;
      }

      // Checking, Storing the inferred information
      auto iter = temp_valueTypesByName.find(n.output(i));
      TypeProto* existingType = nullptr;
      if (iter != temp_valueTypesByName.end()) {
        existingType = iter->second;
        checkShapesAndTypes(*inferred_output_type, *existingType);
      } else {
        // Store the inferred type info in the
        // subgraph temporarily
        auto vi = g.add_value_info();
        vi->set_name(copy_n.output(i));
        existingType = vi->mutable_type();
      }

      if (symbolTable) {
        materializeSymbolicShape(inferred_output_type, *symbolTable);
      }
      mergeShapesAndTypes(*inferred_output_type, existingType);
      if (options.enable_data_propagation && schema->has_data_propagation_function()) {
        DataPropagationContextImpl temp_dataPropagationCtx(
            copy_n, temp_valueTypesByName, temp_initializersByName, *generatedShapeDataByName);
        schema->GetDataPropagationFunction()(temp_dataPropagationCtx);
      }
      // Make merged info available to further inference.
      temp_valueTypesByName[copy_n.output(i)] = existingType;
    }
  }
  for (int i = 0; i < func->output_size(); ++i) {
    const std::string& output_name = func->output().Get(i);
    // Skip if no type inferred for the tensor
    auto hit = temp_valueTypesByName.find(output_name);
    if (hit != temp_valueTypesByName.cend()) {
      if (hit->second->value_case() == TypeProto::kTensorType) {
        // Copy the type info from subgraph to ctx
        // to pass back to maingraph
        auto type = ctx.getOutputType(i)->mutable_tensor_type();
        type->CopyFrom(hit->second->tensor_type());
      } else if (hit->second->value_case() == TypeProto::kSparseTensorType) {
        auto type = ctx.getOutputType(i)->mutable_sparse_tensor_type();
        type->CopyFrom(hit->second->sparse_tensor_type());
      }
    }
  }
}

void InferShapeForFunctionNode(
    const FunctionProto* func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx,
    const ShapeInferenceOptions& options,
    SymbolTable* symbolTable,
    std::unordered_map<std::string, TensorShapeProto>* generatedShapeDataByName) {

  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : func->opset_import()) {
    opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
  }

  InferShapeForFunctionNode(func, opset_imports, schema_registry, ctx, options, symbolTable, generatedShapeDataByName);
}

std::vector<const TypeProto*> GraphInferencerImpl::doInferencing(
    const std::vector<const TypeProto*>& inputTypes,
    const std::vector<const TensorProto*>& inputData) {
  SymbolTable* symbolTable = getSymbolTable();
  int numInputs = int(inputTypes.size());
  std::unordered_set<std::string> initializerNameSet;
  for (const auto& tp : g_->initializer()) {
    initializerNameSet.insert(tp.name());
  }

  if (getIRVersion() >= 4) {
    if (g_->input_size() != numInputs) {
      fail_shape_inference("Graph has ", g_->input_size(), " inputs but ", numInputs, " were provided");
    }
    for (int i = 0; i < g_->input_size(); ++i) {
      if (initializerNameSet.count(g_->input(i).name()) > 0) {
        fail_shape_inference("Cannot use the same name as both a subgraph initializer and subgraph input: ",
          g_->input(i).name());
      }
    }
  } else {
    // IR < 4 requires all initializers to be optional inputs
    // So the number of graph input can be larger than the number of node input 
    if (numInputs > g_->input_size()) {
      fail_shape_inference("Graph has ", g_->input_size(), " inputs but ", numInputs, " were provided.",
        "The number of graph input cannot be smaller than the number of node input" );
    } else if (numInputs < g_->input_size()) {
      for (int i = 0; i < g_->input_size(); ++i) {
        if (i < numInputs && initializerNameSet.count(g_->input(i).name()) > 0) {
          fail_shape_inference("Graph initializer names must appear after the actual inputs: ",
            g_->input(i).name());
        } else if (i >= numInputs && initializerNameSet.count(g_->input(i).name()) == 0) {
          // Further check whether the additional input is in initializers
          fail_shape_inference("Cannot find missing input: ", g_->input(i).name(), "in initializers. ");
        }
      }
    }
  }

  for (int i = 0, end = numInputs; i < end; ++i) {
    const TypeProto* inferredInput = inputTypes[i];

    if (!inferredInput)
      continue;

    TypeProto* graphInput = g_->mutable_input(i)->mutable_type();

    if (inferredInput->value_case() == TypeProto::kTensorType) {
      const auto& inferredType = inferredInput->tensor_type();

      // Bail out early if shape inference does nothing useful.
      if (inferredType.elem_type() == TensorProto::UNDEFINED && !inferredType.has_shape()) {
        continue;
      }
    } else if (inferredInput->value_case() == TypeProto::kSparseTensorType) {
      const auto& inferredType = inferredInput->sparse_tensor_type();

      // Bail out early if shape inference does nothing useful.
      if (inferredType.elem_type() == TensorProto::UNDEFINED && !inferredType.has_shape()) {
        continue;
      }
    }
    // Even if graphInput doesn't have defined type, it will assign inferredType to it
    mergeShapesAndTypes(*inferredInput, graphInput);

    if (symbolTable) {
      materializeSymbolicShape(graphInput, *symbolTable);
    }
  }

  // future: pass inputData into InferShapes either directly, or indirectly by
  // updating initializers that match subgraph inputs.
  (void)inputData;
  ShapeInferenceOptions options {};
  InferShapesImpl(
      g_,
      *context_->outer_scope_value_types_by_name, // never null
      context_->opset_imports,
      options,
      symbolTable,
      context_->schema_registry);

  std::vector<const TypeProto*> graphOutputTypes;
  graphOutputTypes.reserve(g_->output().size());
  for (const ValueInfoProto& output : g_->output()) {
    graphOutputTypes.push_back(&output.type());
  }

  return graphOutputTypes;
}

std::string getErrorWithNodeInfo(NodeProto n, std::runtime_error err) {
  std::string op_name = n.has_name() ? (", node name: " + n.name()) : "";
  return "(op_type:" + n.op_type() + op_name + "): " + err.what();
}

void traverseGraphsToAddExistingSymbols(const GraphProto& g, SymbolTable& symbolTable) {
  symbolTable.addFromGraph(g);
  for (const auto& n : g.node()) {
    for (auto& attr : n.attribute()) {
      if (attr.has_g()) {
        traverseGraphsToAddExistingSymbols(attr.g(), symbolTable);
      }
    }
  }
}

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
