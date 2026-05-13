// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
// onnx/defs/controlflow/old.cc
// NOLINTNEXTLINE(misc-use-internal-linkage)
void ScanInferenceFunction_opset8(InferenceContext& ctx);
// onnx/defs/controlflow/defs.cc
// NOLINTNEXTLINE(misc-use-internal-linkage)
void ScanInferenceFunction(InferenceContext& ctx);
// onnx/defs/controlflow/utils.cc
// NOLINTNEXTLINE(misc-use-internal-linkage)
void ScanVarLenInferenceFunction(InferenceContext& ctx);

namespace Test {

template <class Type>
static void CreateDims(Type& proto, int num_dims) {
  auto mutable_shape = proto.mutable_shape();
  mutable_shape->clear_dim();

  for (int i = 0; i < num_dims; ++i)
    mutable_shape->add_dim();
}

template <class Type>
static void SetDimValues(Type& proto, const std::vector<int>& values) {
  auto mutable_shape = proto.mutable_shape();
  EXPECT_EQ(static_cast<size_t>(mutable_shape->dim_size()), values.size());

  int idx = 0;
  for (auto value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value != -1)
      mutable_dim->set_dim_value(value);
  }
}

template <class Type>
static void SetDimParams(Type& proto, const std::vector<const std::string*>& values) {
  auto mutable_shape = proto.mutable_shape();
  EXPECT_EQ(static_cast<size_t>(mutable_shape->dim_size()), values.size());

  int idx = 0;
  for (const auto* const value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value)
      mutable_dim->set_dim_param(*value);
  }
}

template <class Type>
static void Dump(const Type& t) {
  auto& s_shape = t.shape();
  auto num_dims = s_shape.dim_size();
  std::cout << num_dims << " dims. ";
  for (int i = 0; i < num_dims; ++i) {
    const auto& x = s_shape.dim(0);
    auto y = x.has_dim_value();
    auto z = x.has_dim_param();

    std::cout << "Dim " << i << " Value:" << (y ? ONNX_NAMESPACE::to_string(x.dim_value()) : "<unset>")
              << ", Param:" << (z ? x.dim_param() : "<unset>") << "\n";
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_HasShape) {
  // source has shape, target doesn't
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has no shape, target does
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(target, 1);
    SetDimValues(target, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_EQ(shape.dim_size() == 1 && shape.dim(0).dim_value(), 1);
  }
  // source has shape, target doesn't
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_EQ(shape.dim_size() == 1 && shape.dim(0).dim_value(), 1);
  }

  // source has no shape, target does
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(target, 1);
    SetDimValues(target, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
}
TEST(ShapeInferenceTest, mergeShapeInfo_PreferValueOverParam) {
  std::string param = "A";

  // source has value, target has param. prefer value
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});

    CreateDims(target, 1);
    SetDimParams(target, {&param});

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has param, target has value.
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimParams(source, {&param});

    CreateDims(target, 1);
    SetDimValues(target, {1});

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_EQ(shape.dim_size() == 1 && shape.dim(0).dim_value(), 1);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_CombineShapes) {
  // merge from both sides, preferring real value over -1
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, -1});

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, -1});

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }

  // prefer value over param,
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, 0});
    // replace second dim with a param. the value from the source should be
    // preferred
    const std::string param = "A";
    target.mutable_shape()->mutable_dim(1)->set_dim_param(param);

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, 0});
    // replace second dim with a param. the value from the source should be
    // preferred
    const std::string param = "A";
    target.mutable_shape()->mutable_dim(1)->set_dim_param(param);

    mergeInShapeInfo(source, target);

    Dump(target);
    const auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 && shape.dim(1).dim_value() == 2);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_Mismatches) {
#ifndef ONNX_NO_EXCEPTIONS
  // mismatched num dims
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 3);
    SetDimValues(target, {1, -1, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 3);
    SetDimValues(target, {1, -1, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  // mismatched dim values
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {2, 2});

    CreateDims(target, 2);
    SetDimValues(target, {2, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;

    CreateDims(source, 2);
    SetDimValues(source, {2, 2});

    CreateDims(target, 2);
    SetDimValues(target, {2, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }
#endif
  // mismatched param value. prefer target
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;
    const std::string param_a = "A";
    const std::string param_b = "B";

    CreateDims(source, 1);
    SetDimParams(source, {&param_a});

    CreateDims(target, 1);
    SetDimParams(target, {&param_b});

    mergeInShapeInfo(source, target);

    const auto& shape = target.shape();
    EXPECT_EQ(shape.dim(0).dim_param(), "B");
  }
  {
    TypeProto_SparseTensor source;
    TypeProto_SparseTensor target;
    const std::string param_a = "A";
    const std::string param_b = "B";

    CreateDims(source, 1);
    SetDimParams(source, {&param_a});

    CreateDims(target, 1);
    SetDimParams(target, {&param_b});

    mergeInShapeInfo(source, target);

    const auto& shape = target.shape();
    EXPECT_EQ(shape.dim(0).dim_param(), "B");
  }
}

// Check subgraph inferencing via GraphInferencer using a Scan
static void doInferencingTest(bool use_scan_opset8) {
  OpSchemaRegistry::Instance();
  GraphProto subgraph;

  // simple tensor without shape info
  TypeProto simple_tensor_no_shape;
  auto* tensor_type = simple_tensor_no_shape.mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

  // simple tensor with shape info
  TypeProto simple_tensor = simple_tensor_no_shape;
  simple_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  // setup simple graph that can be used with Scan containing two Identity
  // nodes. one for the loop state variable. one for the scan output.
  {
    NodeProto loop_state_identity;
    loop_state_identity.set_name("loop_state_identity");
    loop_state_identity.set_domain(ONNX_DOMAIN);
    loop_state_identity.set_op_type("Identity");
    loop_state_identity.set_doc_string("loop state identity");
    loop_state_identity.add_input("loop_state_in");
    loop_state_identity.add_output("loop_state_out");

    *subgraph.add_node() = loop_state_identity;

    NodeProto scan_in_out_identity;
    scan_in_out_identity.set_name("scan_in_out_identity");
    scan_in_out_identity.set_domain(ONNX_DOMAIN);
    scan_in_out_identity.set_op_type("Identity");
    scan_in_out_identity.set_doc_string("scan identity");
    scan_in_out_identity.add_input("scan_in");
    scan_in_out_identity.add_output("scan_out");
    *subgraph.add_node() = scan_in_out_identity;

    ValueInfoProto loop_state_in;
    loop_state_in.set_name("loop_state_in");
    *loop_state_in.mutable_type() = simple_tensor;
    *subgraph.add_input() = loop_state_in;

    ValueInfoProto scan_in;
    scan_in.set_name("scan_in");
    *scan_in.mutable_type() = simple_tensor;
    *subgraph.add_input() = scan_in;

    ValueInfoProto loop_state_out = loop_state_in;
    loop_state_out.set_name("loop_state_out");
    *loop_state_out.mutable_type() = simple_tensor_no_shape;
    *subgraph.add_output() = loop_state_out;

    ValueInfoProto scan_state_out = scan_in;
    scan_state_out.set_name("scan_out");
    *scan_state_out.mutable_type() = simple_tensor_no_shape;
    *subgraph.add_output() = scan_state_out;
  }

  std::unordered_map<std::string, int> opset_imports;
  opset_imports[ONNX_DOMAIN] = 8; // Scan is v8

  const std::unordered_map<std::string, TypeProto*> outer_scope_value_types;
  shape_inference::SymbolTableImpl symbolTable;
  symbolTable.addFromGraph(subgraph);
  shape_inference::GraphInferenceContext graphInfCtx(outer_scope_value_types, opset_imports, &symbolTable);
  shape_inference::GraphInferencerImpl graphInferencer(subgraph, graphInfCtx);

  // loop_state_in and scan_in are the two inputs.
  // order in subgraphInputTypes matches their order as graph inputs.
  std::vector<const TypeProto*> subgraphInputTypes = {&simple_tensor, &simple_tensor};

  std::vector<const TensorProto*> subgraphInputData = {};
  ShapeInferenceOptions options{false, 0, false};
  auto output = graphInferencer.doInferencing(subgraphInputTypes, subgraphInputData);

  // check the subgraph outputs had their shape inferred when we called
  // doInferencing directly
  EXPECT_EQ(output.size(), 2);

  auto checkType = [](const TypeProto& type, const TypeProto_Tensor& expect) {
    auto checkDims = [](const TensorShapeProto& l, const TensorShapeProto& r) {
      EXPECT_EQ(l.dim_size(), r.dim_size());

      for (int i = 0, end = l.dim_size(); i < end; ++i) {
        // if (l.dim().Get(i).dim_value() != r.dim().Get(i).dim_value())
        //  break;
        EXPECT_EQ(l.dim().Get(i).dim_value(), r.dim().Get(i).dim_value());
      }
    };

    EXPECT_TRUE(type.has_tensor_type());
    EXPECT_EQ(type.tensor_type().elem_type(), expect.elem_type());
    checkDims(type.tensor_type().shape(), expect.shape());
  };

  checkType(*output[0], simple_tensor.tensor_type());
  checkType(*output[1], simple_tensor.tensor_type());

  // setup Scan node to test subgraph inferencing works as expected when called
  // from the operators type/shape inferencing function
  NodeProto scan;
  {
    AttributeProto num_scan_inputs;
    num_scan_inputs.set_name("num_scan_inputs");
    num_scan_inputs.set_i(1);

    AttributeProto body;
    body.set_name("body");
    *body.mutable_g() = subgraph;

    *scan.add_attribute() = num_scan_inputs;
    *scan.add_attribute() = body;

    scan.set_name("Scan");
    scan.set_domain(ONNX_DOMAIN);
    scan.set_doc_string("Scan node");
    scan.set_op_type("Scan");
    if (use_scan_opset8)
      scan.add_input(""); // optional sequence lens
    scan.add_input("loop_state_start");
    scan.add_input("scan_op_in");
    scan.add_output("loop_state_final");
    scan.add_output("scan_op_out");
  }

  TypeProto loop_state_in_tensor = simple_tensor_no_shape;
  auto* shape = loop_state_in_tensor.mutable_tensor_type()->mutable_shape();
  if (use_scan_opset8)
    shape->add_dim()->set_dim_value(1); // batch size
  shape->add_dim()->set_dim_value(2); // input size. must match subgraph

  TypeProto loop_state_out_tensor = loop_state_in_tensor; // should be unchanged

  TypeProto scan_in_tensor = simple_tensor_no_shape;
  shape = scan_in_tensor.mutable_tensor_type()->mutable_shape();
  if (use_scan_opset8)
    shape->add_dim()->set_dim_value(1); // batch size
  shape->add_dim()->set_dim_value(1); // sequence length
  shape->add_dim()->set_dim_value(2); // input size. must match subgraph

  TypeProto scan_out_tensor = scan_in_tensor; // should be unchanged

  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  valueTypesByName["loop_state_start"] = &loop_state_in_tensor;
  valueTypesByName["scan_op_in"] = &scan_in_tensor;

  shape_inference::InferenceContextImpl ctx(scan, valueTypesByName, {}, {}, options, {}, &graphInfCtx);
  if (use_scan_opset8)
    ScanInferenceFunction_opset8(ctx);
  else
    ScanInferenceFunction(ctx);

  EXPECT_EQ(ctx.getNumOutputs(), 2);
  checkType(*ctx.getOutputType(0), loop_state_out_tensor.tensor_type());
  checkType(*ctx.getOutputType(1), scan_out_tensor.tensor_type());
}

// Check subgraph inferencing via GraphInferencer using a Scan (from opset 8)
TEST(GraphInferencerImplTest, Scan8_BasicTest) {
  doInferencingTest(true);
}

// Check subgraph inferencing via GraphInferencer using a Scan (from opset 9)
TEST(GraphInferencerImplTest, Scan9_BasicTest) {
  doInferencingTest(false);
}

// =========================================================================
// ScanVarLen (opset 27) shape inference tests.
//
// ScanVarLen mirrors Scan but with two key differences:
//   1. An optional non-variadic input `output_lengths` (1-D int64, length K)
//      appears at index 0, followed by a variadic
//      `initial_state_and_scan_inputs` starting at index 1.
//   2. The per-iteration scan-output's dimension at `scan_output_axes[i]`
//      (default 0) is REPLACED with a free/unknown dim in the final output
//      shape, rather than a new sequence dimension being inserted (as Scan
//      does). All other dimensions propagate unchanged from the body output.
// =========================================================================

namespace {

// Builds a body subgraph with two Identity nodes:
//   loop_state_in [N-D, dims=state_dims, elem=state_elem]
//   scan_in_per_iter [M-D, dims=scan_per_iter_dims, elem=scan_per_iter_elem]
// outputs:
//   loop_state_out  (Identity of loop_state_in)
//   scan_out_per_iter (Identity of scan_in_per_iter)
GraphProto MakeScanVarLenBody(
    const std::vector<int>& state_dims,
    int state_elem,
    const std::vector<int>& scan_per_iter_dims,
    int scan_per_iter_elem) {
  GraphProto subgraph;
  subgraph.set_name("scan_var_len_body");

  auto add_identity = [&subgraph](const std::string& in_name, const std::string& out_name) {
    NodeProto node;
    node.set_op_type("Identity");
    node.set_domain(ONNX_DOMAIN);
    node.add_input(in_name);
    node.add_output(out_name);
    *subgraph.add_node() = node;
  };

  add_identity("loop_state_in", "loop_state_out");
  add_identity("scan_in_per_iter", "scan_out_per_iter");

  auto add_value_info =
      [](GraphProto& g, const std::string& name, int elem_type, const std::vector<int>& dims, bool is_input) {
        ValueInfoProto vi;
        vi.set_name(name);
        auto* tt = vi.mutable_type()->mutable_tensor_type();
        tt->set_elem_type(elem_type);
        auto* shape = tt->mutable_shape();
        for (int d : dims) {
          shape->add_dim()->set_dim_value(d);
        }
        if (is_input) {
          *g.add_input() = vi;
        } else {
          *g.add_output() = vi;
        }
      };

  add_value_info(subgraph, "loop_state_in", state_elem, state_dims, /*is_input=*/true);
  add_value_info(subgraph, "scan_in_per_iter", scan_per_iter_elem, scan_per_iter_dims, /*is_input=*/true);
  add_value_info(subgraph, "loop_state_out", state_elem, state_dims, /*is_input=*/false);
  add_value_info(subgraph, "scan_out_per_iter", scan_per_iter_elem, scan_per_iter_dims, /*is_input=*/false);

  return subgraph;
}

// Build a TypeProto for a tensor with the given element type and dims.
TypeProto MakeTensorType(int elem_type, const std::vector<int>& dims) {
  TypeProto type;
  auto* tt = type.mutable_tensor_type();
  tt->set_elem_type(elem_type);
  auto* shape = tt->mutable_shape();
  for (int d : dims) {
    shape->add_dim()->set_dim_value(d);
  }
  return type;
}

// Drive ScanVarLenInferenceFunction with the given body, optional
// output_lengths, optional scan_output_axes attribute, state-var input
// shape/type, and full scan-input shape/type (with sequence axis still
// present). Populates inferred_state_out and inferred_scan_out.
void RunScanVarLenInference(
    const GraphProto& body,
    bool include_output_lengths,
    const std::vector<int64_t>& scan_output_axes,
    const TypeProto& state_in_type,
    const TypeProto& scan_in_type,
    int64_t output_lengths_size,
    TypeProto& inferred_state_out,
    TypeProto& inferred_scan_out) {
  NodeProto node;
  node.set_name("ScanVarLenTestNode");
  node.set_domain(ONNX_DOMAIN);
  node.set_op_type("ScanVarLen");

  // Optional output_lengths input slot (index 0).
  if (include_output_lengths) {
    node.add_input("output_lengths");
  } else {
    node.add_input("");
  }
  node.add_input("state_in");
  node.add_input("scan_in_full");

  node.add_output("state_out");
  node.add_output("scan_out");

  AttributeProto num_scan_inputs_attr;
  num_scan_inputs_attr.set_name("num_scan_inputs");
  num_scan_inputs_attr.set_type(AttributeProto::INT);
  num_scan_inputs_attr.set_i(1);
  *node.add_attribute() = num_scan_inputs_attr;

  AttributeProto body_attr;
  body_attr.set_name("body");
  body_attr.set_type(AttributeProto::GRAPH);
  *body_attr.mutable_g() = body;
  *node.add_attribute() = body_attr;

  if (!scan_output_axes.empty()) {
    AttributeProto axes_attr;
    axes_attr.set_name("scan_output_axes");
    axes_attr.set_type(AttributeProto::INTS);
    for (int64_t v : scan_output_axes) {
      axes_attr.add_ints(v);
    }
    *node.add_attribute() = axes_attr;
  }

  TypeProto output_lengths_type = MakeTensorType(TensorProto::INT64, {static_cast<int>(output_lengths_size)});
  TypeProto state_in_type_copy = state_in_type;
  TypeProto scan_in_type_copy = scan_in_type;

  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  if (include_output_lengths) {
    valueTypesByName["output_lengths"] = &output_lengths_type;
  }
  valueTypesByName["state_in"] = &state_in_type_copy;
  valueTypesByName["scan_in_full"] = &scan_in_type_copy;

  std::unordered_map<std::string, int> opset_imports;
  opset_imports[ONNX_DOMAIN] = 27;

  const std::unordered_map<std::string, TypeProto*> outer_scope_value_types;
  shape_inference::SymbolTableImpl symbolTable;
  shape_inference::GraphInferenceContext graphInfCtx(outer_scope_value_types, opset_imports, &symbolTable);

  ShapeInferenceOptions options{false, 0, false};
  shape_inference::InferenceContextImpl ctx(node, valueTypesByName, {}, {}, options, {}, &graphInfCtx);

  ScanVarLenInferenceFunction(ctx);

  ASSERT_EQ(ctx.getNumOutputs(), 2u);
  inferred_state_out = *ctx.getOutputType(0);
  inferred_scan_out = *ctx.getOutputType(1);
}

void ExpectDimValue(const TensorShapeProto& shape, int axis, int expected_value) {
  ASSERT_LT(axis, shape.dim_size()) << "axis " << axis << " out of bounds for rank " << shape.dim_size();
  const auto& dim = shape.dim(axis);
  ASSERT_TRUE(dim.has_dim_value()) << "expected concrete dim at axis " << axis;
  EXPECT_EQ(dim.dim_value(), expected_value) << "at axis " << axis;
}

void ExpectFreeDim(const TensorShapeProto& shape, int axis) {
  ASSERT_LT(axis, shape.dim_size()) << "axis " << axis << " out of bounds for rank " << shape.dim_size();
  const auto& dim = shape.dim(axis);
  EXPECT_FALSE(dim.has_dim_value()) << "expected free/unknown dim at axis " << axis
                                    << " but got dim_value=" << dim.dim_value();
  EXPECT_FALSE(dim.has_dim_param()) << "expected free/unknown dim at axis " << axis << " but got dim_param='"
                                    << dim.dim_param() << "'";
}

} // namespace

// Case 1: basic 1-state + 1-scan-input + 1-scan-output, default axes.
// State var: float[3] -> float[3] (propagated 1:1).
// Scan input: float[T=5, 4] -> per-iter float[4] -> scan output float[?]
// (the dim at scan_output_axes[0]=0 is replaced with a free dim).
TEST(ShapeInferenceTest, ScanVarLen27_BasicTest) {
  GraphProto body =
      MakeScanVarLenBody(/*state_dims=*/{3}, TensorProto::FLOAT, /*scan_per_iter_dims=*/{4}, TensorProto::FLOAT);
  TypeProto state_in = MakeTensorType(TensorProto::FLOAT, {3});
  TypeProto scan_in_full = MakeTensorType(TensorProto::FLOAT, {5, 4}); // T=5, D=4

  TypeProto state_out, scan_out;
  RunScanVarLenInference(
      body,
      /*include_output_lengths=*/false,
      /*scan_output_axes=*/{},
      state_in,
      scan_in_full,
      /*output_lengths_size=*/1,
      state_out,
      scan_out);

  // State var output: type and shape preserved exactly.
  ASSERT_TRUE(state_out.has_tensor_type());
  EXPECT_EQ(state_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(state_out.tensor_type().shape().dim_size(), 1);
  ExpectDimValue(state_out.tensor_type().shape(), 0, 3);

  // Scan output: rank == body output rank (= 1); axis 0 is free; element type
  // propagated from body subgraph.
  ASSERT_TRUE(scan_out.has_tensor_type());
  EXPECT_EQ(scan_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(scan_out.tensor_type().shape().dim_size(), 1);
  ExpectFreeDim(scan_out.tensor_type().shape(), 0);
}

// Case 2: scan_output_axes != 0. With a 2-D body scan output of shape [4, 6]
// and scan_output_axes=[1], the final scan output should be [4, ?] (axis 0
// preserved, axis 1 replaced with a free dim).
TEST(ShapeInferenceTest, ScanVarLen27_ScanOutputAxesNonZero) {
  GraphProto body = MakeScanVarLenBody(
      /*state_dims=*/{3}, TensorProto::FLOAT, /*scan_per_iter_dims=*/{4, 6}, TensorProto::FLOAT);
  TypeProto state_in = MakeTensorType(TensorProto::FLOAT, {3});
  TypeProto scan_in_full = MakeTensorType(TensorProto::FLOAT, {5, 4, 6}); // T=5 along axis 0

  TypeProto state_out, scan_out;
  RunScanVarLenInference(
      body,
      /*include_output_lengths=*/false,
      /*scan_output_axes=*/{1},
      state_in,
      scan_in_full,
      /*output_lengths_size=*/1,
      state_out,
      scan_out);

  // State var unchanged.
  ASSERT_TRUE(state_out.has_tensor_type());
  EXPECT_EQ(state_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(state_out.tensor_type().shape().dim_size(), 1);
  ExpectDimValue(state_out.tensor_type().shape(), 0, 3);

  // Scan output: rank 2, axis 0 preserved as 4, axis 1 is the free concat axis.
  ASSERT_TRUE(scan_out.has_tensor_type());
  EXPECT_EQ(scan_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(scan_out.tensor_type().shape().dim_size(), 2);
  ExpectDimValue(scan_out.tensor_type().shape(), 0, 4);
  ExpectFreeDim(scan_out.tensor_type().shape(), 1);
}

// Case 3: optional output_lengths input present. Inference should accept the
// 1-D int64 tensor of length K (= num scan outputs) and produce the same shapes
// as the basic case.
TEST(ShapeInferenceTest, ScanVarLen27_WithOutputLengths) {
  GraphProto body =
      MakeScanVarLenBody(/*state_dims=*/{3}, TensorProto::FLOAT, /*scan_per_iter_dims=*/{4}, TensorProto::FLOAT);
  TypeProto state_in = MakeTensorType(TensorProto::FLOAT, {3});
  TypeProto scan_in_full = MakeTensorType(TensorProto::FLOAT, {5, 4});

  TypeProto state_out, scan_out;
  RunScanVarLenInference(
      body,
      /*include_output_lengths=*/true,
      /*scan_output_axes=*/{},
      state_in,
      scan_in_full,
      /*output_lengths_size=*/1,
      state_out,
      scan_out);

  // Identical to the basic case.
  ASSERT_TRUE(state_out.has_tensor_type());
  EXPECT_EQ(state_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(state_out.tensor_type().shape().dim_size(), 1);
  ExpectDimValue(state_out.tensor_type().shape(), 0, 3);

  ASSERT_TRUE(scan_out.has_tensor_type());
  EXPECT_EQ(scan_out.tensor_type().elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(scan_out.tensor_type().shape().dim_size(), 1);
  ExpectFreeDim(scan_out.tensor_type().shape(), 0);
}

// Case 4: the scan-output element type is taken from the body subgraph output,
// not from any op input. Use a body whose scan output is INT32 while the scan
// input is FLOAT.
TEST(ShapeInferenceTest, ScanVarLen27_ScanOutputElemTypeFromBody) {
  GraphProto body = MakeScanVarLenBody(
      /*state_dims=*/{3}, TensorProto::FLOAT, /*scan_per_iter_dims=*/{4}, TensorProto::INT32);
  TypeProto state_in = MakeTensorType(TensorProto::FLOAT, {3});
  // Scan input full tensor element type must match the body subgraph's
  // per-iter scan input type (INT32) since the body identity-copies it.
  TypeProto scan_in_full = MakeTensorType(TensorProto::INT32, {5, 4});

  TypeProto state_out, scan_out;
  RunScanVarLenInference(
      body,
      /*include_output_lengths=*/false,
      /*scan_output_axes=*/{},
      state_in,
      scan_in_full,
      /*output_lengths_size=*/1,
      state_out,
      scan_out);

  // State var output element type comes from state input.
  ASSERT_TRUE(state_out.has_tensor_type());
  EXPECT_EQ(state_out.tensor_type().elem_type(), TensorProto::FLOAT);

  // Scan output element type comes from the body subgraph (INT32).
  ASSERT_TRUE(scan_out.has_tensor_type());
  EXPECT_EQ(scan_out.tensor_type().elem_type(), TensorProto::INT32);
  ASSERT_EQ(scan_out.tensor_type().shape().dim_size(), 1);
  ExpectFreeDim(scan_out.tensor_type().shape(), 0);
}

static void ParseAndInfer(ModelProto& model, const char* modelStr) {
  OnnxParser parser(modelStr);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  ShapeInferenceOptions options{true, 1, true};
  ONNX_NAMESPACE::shape_inference::InferShapes(model, ONNX_NAMESPACE::OpSchemaRegistry::Instance(), options);
}

static void RunReshapeShapeInfTest(const char* modelStr, TensorShapeProto& expectedShape) {
  ModelProto model;
  ParseAndInfer(model, modelStr);

  const auto inferredShape = model.graph().output(0).type().tensor_type().shape();
  EXPECT_EQ(inferredShape.dim_size(), expectedShape.dim_size());

  for (int i = 0; i < inferredShape.dim_size(); i++) {
    EXPECT_TRUE(
        (inferredShape.dim(i).has_dim_value() && expectedShape.dim(i).has_dim_value()) ||
        (inferredShape.dim(i).has_dim_param() && expectedShape.dim(i).has_dim_param()));

    EXPECT_TRUE(
        inferredShape.dim(i).has_dim_value() ? inferredShape.dim(i).dim_value() == expectedShape.dim(i).dim_value()
                                             : inferredShape.dim(i).dim_param() == expectedShape.dim(i).dim_param());
  }
}
TEST(ShapeInferenceTest, ReshapeTestWithShapeAsSymInput) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[batch_size, 256, 768, 3] x, float[batch_size, 196608] m) => (float[?, ?, ?] z)
{
    y = Shape<start = 0, end = 3>(x)
    z = Reshape(m, y)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_param("batch_size");
  expectedShape.mutable_dim()->Add()->set_dim_value(256);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, ReshapeTestWithShapeAsInitializer) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[1, 196608] m) => (float[?, ?, ?] z)
<int64[3] shape = {1, 768, 256}>
{
    z = Reshape(m, shape)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_value(1);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);
  expectedShape.mutable_dim()->Add()->set_dim_value(256);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, ReshapeTestWithShapeAsInitializer1) {
  const char* modelStr = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 15],
  producer_name: "DataPropagationTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for data propagation."
>
agraph (float[1, 196608] m) => (float[?, ?, ?] z)
<int64[3] shape = {1, -1, 256}>
{
    z = Reshape(m, shape)
}
)ONNX";

  TensorShapeProto expectedShape;
  expectedShape.mutable_dim()->Add()->set_dim_value(1);
  expectedShape.mutable_dim()->Add()->set_dim_value(768);
  expectedShape.mutable_dim()->Add()->set_dim_value(256);

  RunReshapeShapeInfTest(modelStr, expectedShape);
}

TEST(ShapeInferenceTest, CheckShapesAndTypesTest) {
#ifndef ONNX_NO_EXCEPTIONS
  // Tensor element types mismatch should cause an exception.
  TypeProto tensor_infer;
  auto* tensor_infer_type = tensor_infer.mutable_tensor_type();
  tensor_infer_type->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto tensor_exist;
  auto* tensor_exist_type = tensor_exist.mutable_tensor_type();
  tensor_exist_type->set_elem_type(TensorProto_DataType_UINT8);

  EXPECT_THROW(shape_inference::checkShapesAndTypes(tensor_infer, tensor_exist), ONNX_NAMESPACE::InferenceError);
#endif
}

TEST(ShapeInferenceTest, CustomOpTest) {
  const char* modelStr = R"ONNX(
<ir_version: 8,  opset_import: ["" : 15, "custom.domain" : 1]>
agraph (float[256, 768, 3] x) => (z1, z2)
{
    z1 = custom.domain.CustomOp (x)
    # Inference cannot determine the type/shape of z1
    z2 = Abs(x)
    # Inference SHOULD determine the type/shape of z2 (same as that of x)
}
)ONNX";

  ModelProto model;
  ParseAndInfer(model, modelStr);

  const auto& z1_value_info = model.graph().output(0);
  // Check no inferred type for z1 (It's a quirk of the implementation that it
  // has a dummy TypeProto, but it should have no values filled in.)
  ASSERT_TRUE(z1_value_info.has_type());
  ASSERT_FALSE(z1_value_info.type().has_tensor_type());

  // Check inferred type for z2:
  const auto& z2_value_info = model.graph().output(1);
  ASSERT_TRUE(z2_value_info.has_type());
  ASSERT_TRUE(z2_value_info.type().has_tensor_type());
  EXPECT_EQ(z2_value_info.type().tensor_type().elem_type(), TensorProto_DataType_FLOAT);
  EXPECT_EQ(z2_value_info.type().tensor_type().shape().dim_size(), 3);
  EXPECT_EQ(z2_value_info.type().tensor_type().shape().dim(0).dim_value(), 256);
  EXPECT_EQ(z2_value_info.type().tensor_type().shape().dim(1).dim_value(), 768);
  EXPECT_EQ(z2_value_info.type().tensor_type().shape().dim(2).dim_value(), 3);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
