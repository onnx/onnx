#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
namespace shape_inference {
void checkShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    const TypeProto_Tensor& existingType) {
  if (inferredType.elem_type() != TensorProto::UNDEFINED &&
      existingType.elem_type() != TensorProto::UNDEFINED &&
      existingType.elem_type() != inferredType.elem_type()) {
    std::stringstream ss;
    ss << "Inferred elem type differs from existing elem type: ("
       << inferredType.elem_type() << ") vs (" << existingType.elem_type()
       << ")";
    throw std::runtime_error(ss.str());
  }

  if (!inferredType.has_shape() || !existingType.has_shape()) {
    return;
  }

  if (inferredType.shape().dim_size() != existingType.shape().dim_size()) {
    std::stringstream ss;
    ss << "Inferred shape and existing shape differ in rank: ("
       << inferredType.shape().dim_size() << ") vs ("
       << existingType.shape().dim_size() << ")";
    throw std::runtime_error(ss.str());
  }

  for (int i = 0; i < inferredType.shape().dim_size(); ++i) {
    const auto& inferredDim = inferredType.shape().dim(i);
    const auto& existingDim = existingType.shape().dim(i);
    if (inferredDim.has_dim_value() && existingDim.has_dim_value() &&
        inferredDim.dim_value() != existingDim.dim_value()) {
      std::stringstream ss;
      ss << "Inferred shape and existing shape differ in dimension " << i
         << ": (" << inferredDim.dim_value() << ") vs ("
         << existingDim.dim_value() << ")";
      throw std::runtime_error(ss.str());
    }
  }
}

void mergeShapesAndTypes(
    const TypeProto_Tensor& inferredType,
    TypeProto_Tensor* existingType) {
  if (inferredType.elem_type() != TensorProto::UNDEFINED) {
    if (existingType->elem_type() == TensorProto::UNDEFINED) {
      existingType->set_elem_type(inferredType.elem_type());
    } else if (existingType->elem_type() != inferredType.elem_type()) {
      fail_type_inference(
          "type mismatch. existing=",
          existingType->elem_type(),
          " inferred=",
          inferredType.elem_type());
    }
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

static void InferShapesImpl(
    GraphProto* g,
    const std::unordered_map<std::string, TypeProto*>&
        outer_scope_value_types_by_name,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    const IFunctionBuilderRegistry* func_registry =
        &FunctionBuilderRegistry::OnnxInstance()) {
  std::unordered_map<std::string, TypeProto*> valueTypesByName{
      outer_scope_value_types_by_name};

  GraphInferenceContext graphInferenceContext{
      valueTypesByName, opset_imports, schema_registry, func_registry};

  for (auto& vi : *g->mutable_value_info()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }
  for (auto& vi : *g->mutable_input()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }
  for (auto& vi : *g->mutable_output()) {
    if (vi.has_type())
      valueTypesByName[vi.name()] = vi.mutable_type();
  }

  std::unordered_map<std::string, const TensorProto*> inputDataByName;
  for (const auto& tp : g->initializer()) {
    inputDataByName[tp.name()] = &tp;
  }
  // Collect data from constant nodes.
  for (const auto& n : g->node()) {
      if (n.op_type() != "Constant" || n.output().size() != 1) {
          continue;
      }
      for (const auto& attr : n.attribute()) {
          if (attr.name() == "value" &&
              attr.type() == AttributeProto::TENSOR &&
              attr.has_t()) {
              inputDataByName[n.output(0)] = &attr.t();
          }
      }
  }

  for (auto& n : *g->mutable_node()) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      continue;
    }
    auto domain_version = dit->second;

    const auto schema =
        schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    InferenceContextImpl ctx(
        n, valueTypesByName, inputDataByName, &graphInferenceContext);
    if (!schema) {
      if (nullptr == func_registry) {
        continue;
      }
      // The node is not referring a primitive operator.
      // Check whether it's referring to a function.
      // If it's referring to a function.
      auto func =
          func_registry->GetFunction(n.op_type(), domain_version, n.domain());
      if (nullptr == func) {
        continue;
      }
      try {
        InferShapeForFunctionNode(*func, schema_registry, ctx);
      } catch (const ONNX_NAMESPACE::InferenceError& ex) {
        (void)ex;
        // Continue with inference for remaining nodes
        continue;
      }
    } else {
      try {
        schema->GetTypeAndShapeInferenceFunction()(ctx);
      } catch (const ONNX_NAMESPACE::InferenceError& ex) {
        (void)ex;
        // Continue with inference for remaining nodes
        continue;
      }
    }

    try {
      for (int i = 0; i < n.output_size(); ++i) {
        if (!ctx.getOutputType(i)->has_tensor_type()) {
          continue;
        }
        const auto& inferredType = ctx.getOutputType(i)->tensor_type();

        // Bail out early if shape inference does nothing useful.
        if (inferredType.elem_type() == TensorProto::UNDEFINED &&
            !inferredType.has_shape()) {
          continue;
        }

        // Find any pre-existing type and shape info. If there is such,
        // then check for compatibility with the inferred
        // information. Otherwise, initialize it in an empty state.
        auto iter = valueTypesByName.find(n.output(i));
        TypeProto* existingType = nullptr;
        if (iter != valueTypesByName.end()) {
          existingType = iter->second;
          checkShapesAndTypes(inferredType, existingType->tensor_type());
        } else {
          auto vi = g->add_value_info();
          vi->set_name(n.output(i));
          existingType = vi->mutable_type();
        }

        // Now we can merge pre-existing and inferred info, without
        // further need for error-checking.
        mergeShapesAndTypes(inferredType, existingType->mutable_tensor_type());

        // Make merged info available to further inference.
        valueTypesByName[n.output(i)] = existingType;
      }
    } catch (const std::runtime_error& err) {
      std::string op_name = n.has_name() ? n.name() : "no name";
      std::cerr << "(op_type:" << n.op_type() << ", name:" << n.name() << "): " << err.what() << '\n';
      throw;
    }
  }
}

void InferShapes(
    GraphProto* g,
    const std::unordered_map<std::string, int>& opset_imports,
    const ISchemaRegistry* schema_registry,
    const IFunctionBuilderRegistry* func_registry) {
  InferShapesImpl(
      g,
      std::unordered_map<std::string, TypeProto*>(0),
      opset_imports,
      schema_registry,
      func_registry);
}

void InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry,
    const IFunctionBuilderRegistry* func_registry) {
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : m.opset_import()) {
    opset_imports[opset_import.domain()] =
        static_cast<int>(opset_import.version());
  }
  auto* g = m.mutable_graph();
  InferShapesImpl(
      g,
      std::unordered_map<std::string, TypeProto*>(0),
      opset_imports,
      schema_registry,
      func_registry);
}

void InferShapeForFunctionNode(
    const FunctionProto& func,
    const ISchemaRegistry* schema_registry,
    InferenceContext& ctx) {
  int domain_version = (int)func.since_version();
  GraphProto g;
  // Get a temporary tensor-shape map
  std::unordered_map<std::string, TypeProto*> temp_valueTypesByName;
  std::vector<TypeProto> temp_types_cache(func.input_size());
  for (int i = 0; i < func.input_size(); ++i) {
    temp_types_cache[i] = *ctx.getInputType(i);
    temp_valueTypesByName[func.input().Get(i)] = &temp_types_cache.back();
  }
  // Get a temporary initial value map
  std::unordered_map<std::string, const TensorProto*> temp_initializersByName;
  for (int i = 0; i < static_cast<int>(ctx.getNumInputs()); ++i) {
    if (ctx.getInputData(i) != nullptr && i < func.input_size()) {
      temp_initializersByName[func.input().Get(i)] = ctx.getInputData(i);
    }
  }
  std::unordered_map<std::string, const AttributeProto*> attr_map;
  for (auto& attr : func.attribute()) {
    if (ctx.getAttribute(attr) != nullptr) {
      attr_map[attr] = ctx.getAttribute(attr);
    }
  }

  for (auto& n : func.node()) {
    const auto schema =
        schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    if (!schema) {
      return;
    }
    NodeProto copy_n(n);
    // Add attribute information into the temporary node
    copy_n.clear_attribute();
    for (auto attr : n.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          copy_n.add_attribute()->CopyFrom(*attr_map[attr.ref_attr_name()]);
        }
      } else {
        copy_n.add_attribute()->CopyFrom(attr);
      }
    }

    InferenceContextImpl temp_ctx(
        copy_n, temp_valueTypesByName, temp_initializersByName);
    schema->GetTypeAndShapeInferenceFunction()(temp_ctx);
    for (int i = 0; i < copy_n.output_size(); ++i) {
      if (!temp_ctx.getOutputType(i)->has_tensor_type()) {
        continue;
      }
      const auto& inferredType = temp_ctx.getOutputType(i)->tensor_type();

      // Bail out early if shape inference does nothing useful.
      if (inferredType.elem_type() == TensorProto::UNDEFINED &&
          !inferredType.has_shape()) {
        continue;
      }

      // Checking, Storing the inferred information
      auto iter = temp_valueTypesByName.find(n.output(i));
      TypeProto* existingType = nullptr;
      if (iter != temp_valueTypesByName.end()) {
        existingType = iter->second;
        checkShapesAndTypes(inferredType, existingType->tensor_type());
      } else {
        // Store the inferred type info in the
        // subgraph temporarily
        auto vi = g.add_value_info();
        vi->set_name(copy_n.output(i));
        existingType = vi->mutable_type();
      }
      mergeShapesAndTypes(inferredType, existingType->mutable_tensor_type());
      // Make merged info available to further inference.
      temp_valueTypesByName[copy_n.output(i)] = existingType;
    }
  }
  for (int i = 0; i < func.output_size(); ++i) {
    std::string output_name = func.output().Get(i);
    // Skip if no type inferred for the tensor
    if (!temp_valueTypesByName.count(output_name)) {
      continue;
    }
    // Copy the type info from subgraph to ctx
    // to pass back to maingraph
    auto type = ctx.getOutputType(i)->mutable_tensor_type();
    type->CopyFrom(temp_valueTypesByName[output_name]->tensor_type());
  }
}

std::vector<const TypeProto*> GraphInferencerImpl::doInferencing(
    const std::vector<const TypeProto*>& inputTypes,
    const std::vector<const TensorProto*>& inputData) {
  int numInputs = int(inputTypes.size());

  if (g_->input_size() != numInputs)
    fail_shape_inference(
        "Graph has ",
        g_->input_size(),
        " inputs but ",
        numInputs,
        " were provided");

  for (int i = 0, end = numInputs; i < end; ++i) {
    const TypeProto* inferredInput = inputTypes[i];

    if (!inferredInput)
      continue;

    TypeProto* graphInput = g_->mutable_input(i)->mutable_type();

    if (!graphInput->has_tensor_type()) {
      continue;
    }

    if (!inferredInput->has_tensor_type())
      fail_type_inference(
          "Graph input #",
          i,
          " is tensor type, but provided type is ",
          inferredInput->value_case());

    const auto& inferredType = inferredInput->tensor_type();

    // Bail out early if shape inference does nothing useful.
    if (inferredType.elem_type() == TensorProto::UNDEFINED &&
        !inferredType.has_shape()) {
      continue;
    }

    mergeShapesAndTypes(inferredType, graphInput->mutable_tensor_type());
  }

  // future: pass inputData into InferShapes either directly, or indirectly by
  // updating initializers that match subgraph inputs.
  (void)inputData;

  InferShapesImpl(
      g_,
      *context_->outer_scope_value_types_by_name, // never null
      context_->opset_imports,
      context_->schema_registry,
      context_->func_registry);

  std::vector<const TypeProto*> graphOutputTypes;
  for (const ValueInfoProto& output : g_->output()) {
    graphOutputTypes.push_back(&output.type());
  }

  return graphOutputTypes;
}

} // namespace shape_inference
} // namespace ONNX_NAMESPACE
