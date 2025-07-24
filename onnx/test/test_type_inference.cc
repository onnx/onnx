#include "gtest/gtest.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/proto_utils.h"
#include "onnx/checker.h"
#include <memory>

using namespace ONNX_NAMESPACE;

class TypeInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Common cleanup for tests
    }

    // Helper to create a simple Add model
    ModelProto CreateSimpleAddModel() {
        ModelProto model;
        auto* graph = model.mutable_graph();
        graph->set_name("test_graph");

        // Create inputs
        auto* input1 = graph->add_input();
        input1->set_name("input1");
        auto* input1_type = input1->mutable_type()->mutable_tensor_type();
        input1_type->set_elem_type(TensorProto::FLOAT);
        input1_type->mutable_shape()->add_dim()->set_dim_value(2);
        input1_type->mutable_shape()->add_dim()->set_dim_value(3);

        auto* input2 = graph->add_input();
        input2->set_name("input2");
        auto* input2_type = input2->mutable_type()->mutable_tensor_type();
        input2_type->set_elem_type(TensorProto::FLOAT);
        input2_type->mutable_shape()->add_dim()->set_dim_value(2);
        input2_type->mutable_shape()->add_dim()->set_dim_value(3);

        // Create output
        auto* output = graph->add_output();
        output->set_name("output");
        auto* output_type = output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(TensorProto::FLOAT);
        output_type->mutable_shape()->add_dim()->set_dim_value(2);
        output_type->mutable_shape()->add_dim()->set_dim_value(3);

        // Create Add node
        auto* node = graph->add_node();
        node->set_op_type("Add");
        node->add_input("input1");
        node->add_input("input2");
        node->add_output("output");

        return model;
    }

    // Helper to create a model with initializers
    ModelProto CreateModelWithInitializer() {
        ModelProto model;
        auto* graph = model.mutable_graph();
        graph->set_name("test_graph_with_init");

        // Create input
        auto* input = graph->add_input();
        input->set_name("input");
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(TensorProto::FLOAT);
        input_type->mutable_shape()->add_dim()->set_dim_value(2);
        input_type->mutable_shape()->add_dim()->set_dim_value(3);

        // Create initializer
        auto* init = graph->add_initializer();
        init->set_name("weight");
        init->set_data_type(TensorProto::FLOAT);
        init->add_dims(3);
        init->add_dims(4);
        // Add some dummy data
        for (int i = 0; i < 12; ++i) {
            init->add_float_data(1.0f);
        }

        // Create output
        auto* output = graph->add_output();
        output->set_name("output");
        auto* output_type = output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(TensorProto::FLOAT);
        output_type->mutable_shape()->add_dim()->set_dim_value(2);
        output_type->mutable_shape()->add_dim()->set_dim_value(4);

        // Create MatMul node
        auto* node = graph->add_node();
        node->set_op_type("MatMul");
        node->add_input("input");
        node->add_input("weight");
        node->add_output("output");

        return model;
    }

    // Helper to create multi-node model
    ModelProto CreateMultiNodeModel() {
        ModelProto model;
        auto* graph = model.mutable_graph();
        graph->set_name("multi_node_graph");

        // Create inputs
        auto* input1 = graph->add_input();
        input1->set_name("input1");
        auto* input1_type = input1->mutable_type()->mutable_tensor_type();
        input1_type->set_elem_type(TensorProto::FLOAT);
        input1_type->mutable_shape()->add_dim()->set_dim_value(2);
        input1_type->mutable_shape()->add_dim()->set_dim_value(3);

        auto* input2 = graph->add_input();
        input2->set_name("input2");
        auto* input2_type = input2->mutable_type()->mutable_tensor_type();
        input2_type->set_elem_type(TensorProto::FLOAT);
        input2_type->mutable_shape()->add_dim()->set_dim_value(2);
        input2_type->mutable_shape()->add_dim()->set_dim_value(3);

        // Create final output
        auto* output = graph->add_output();
        output->set_name("final_output");
        auto* output_type = output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(TensorProto::FLOAT);
        output_type->mutable_shape()->add_dim()->set_dim_value(2);
        output_type->mutable_shape()->add_dim()->set_dim_value(3);

        // Create Add node
        auto* add_node = graph->add_node();
        add_node->set_op_type("Add");
        add_node->add_input("input1");
        add_node->add_input("input2");
        add_node->add_output("intermediate");

        // Create Relu node
        auto* relu_node = graph->add_node();
        relu_node->set_op_type("Relu");
        relu_node->add_input("intermediate");
        relu_node->add_output("final_output");

        return model;
    }
};

TEST_F(TypeInferenceTest, BasicTypeInference) {
    ModelProto model = CreateSimpleAddModel();
    
    // Clear value_info to test inference
    model.mutable_graph()->clear_value_info();
    
    // Perform type inference
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
    
    // Check that the model is still valid
    ASSERT_NO_THROW(checker::check_model(model));
}

TEST_F(TypeInferenceTest, TypeInferenceWithInitializers) {
    ModelProto model = CreateModelWithInitializer();
    model.mutable_graph()->clear_value_info();
    
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
    ASSERT_NO_THROW(checker::check_model(model));
}

TEST_F(TypeInferenceTest, MultiNodeTypeInference) {
    ModelProto model = CreateMultiNodeModel();
    model.mutable_graph()->clear_value_info();
    
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
    
    // Check that intermediate value type was inferred
    bool intermediate_found = false;
    for (const auto& value_info : model.graph().value_info()) {
        if (value_info.name() == "intermediate") {
            intermediate_found = true;
            EXPECT_EQ(value_info.type().tensor_type().elem_type(), TensorProto::FLOAT);
        }
    }
    EXPECT_TRUE(intermediate_found) << "Intermediate value type should be inferred";
}

TEST_F(TypeInferenceTest, StrictMode) {
    ModelProto model = CreateSimpleAddModel();
    
    // Test that strict mode doesn't throw for valid models
    ASSERT_NO_THROW(InferTypes(model, false, true, false));
}

TEST_F(TypeInferenceTest, CheckTypeFlag) {
    ModelProto model = CreateSimpleAddModel();
    
    // Test with check_type=true
    ASSERT_NO_THROW(InferTypes(model, true, false, false));
}

TEST_F(TypeInferenceTest, DataPropFlag) {
    ModelProto model = CreateSimpleAddModel();
    
    // Test with data_prop=true
    ASSERT_NO_THROW(InferTypes(model, false, false, true));
}

TEST_F(TypeInferenceTest, InferTypesForNode) {
    ModelProto model = CreateSimpleAddModel();
    const auto& node = model.graph().node(0);
    
    // Build input types map
    std::unordered_map<std::string, TypeProto> input_types;
    for (const auto& input : model.graph().input()) {
        input_types[input.name()] = input.type();
    }
    
    // Test node-level type inference
    std::unordered_map<std::string, TypeProto> output_types;
    ASSERT_NO_THROW(InferTypesForNode(node, input_types, output_types, false));
    
    // Check that output was inferred
    EXPECT_GT(output_types.size(), 0);
    if (output_types.find("output") != output_types.end()) {
        EXPECT_EQ(output_types["output"].tensor_type().elem_type(), TensorProto::FLOAT);
    }
}

TEST_F(TypeInferenceTest, ValidateTypeConsistency) {
    ModelProto model = CreateSimpleAddModel();
    
    // Build value types map
    std::unordered_map<std::string, TypeProto> value_types;
    for (const auto& input : model.graph().input()) {
        value_types[input.name()] = input.type();
    }
    for (const auto& output : model.graph().output()) {
        value_types[output.name()] = output.type();
    }
    
    // Test type consistency validation
    ASSERT_NO_THROW(ValidateTypeConsistency(model.graph(), value_types));
}

TEST_F(TypeInferenceTest, EmptyModel) {
    ModelProto model;
    auto* graph = model.mutable_graph();
    graph->set_name("empty_graph");
    
    // Should not crash on empty model
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
}

TEST_F(TypeInferenceTest, UnknownOperator) {
    ModelProto model;
    auto* graph = model.mutable_graph();
    graph->set_name("unknown_op_graph");
    
    // Create input
    auto* input = graph->add_input();
    input->set_name("input");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(TensorProto::FLOAT);
    input_type->mutable_shape()->add_dim()->set_dim_value(2);
    input_type->mutable_shape()->add_dim()->set_dim_value(3);
    
    // Create output
    auto* output = graph->add_output();
    output->set_name("output");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(TensorProto::FLOAT);
    output_type->mutable_shape()->add_dim()->set_dim_value(2);
    output_type->mutable_shape()->add_dim()->set_dim_value(3);
    
    // Create node with unknown operator
    auto* node = graph->add_node();
    node->set_op_type("UnknownOp");
    node->add_input("input");
    node->add_output("output");
    
    // Should not crash in non-strict mode
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
}

TEST_F(TypeInferenceTest, PreserveExistingValueInfo) {
    ModelProto model = CreateSimpleAddModel();
    
    // Add some existing value_info
    auto* value_info = model.mutable_graph()->add_value_info();
    value_info->set_name("existing_info");
    auto* type = value_info->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::INT32);
    
    size_t original_count = model.graph().value_info_size();
    
    InferTypes(model, false, false, false);
    
    // Should have at least the original value_info entries
    EXPECT_GE(model.graph().value_info_size(), original_count);
    
    // Check that existing info is preserved
    bool existing_found = false;
    for (const auto& vi : model.graph().value_info()) {
        if (vi.name() == "existing_info") {
            existing_found = true;
            EXPECT_EQ(vi.type().tensor_type().elem_type(), TensorProto::INT32);
        }
    }
    EXPECT_TRUE(existing_found) << "Existing value_info should be preserved";
}

// Test for different tensor types
class TypeInferenceParameterizedTest : public TypeInferenceTest,
                                      public ::testing::WithParamInterface<TensorProto::DataType> {};

TEST_P(TypeInferenceParameterizedTest, DifferentTensorTypes) {
    TensorProto::DataType data_type = GetParam();
    
    ModelProto model;
    auto* graph = model.mutable_graph();
    graph->set_name("parameterized_test");
    
    // Create inputs with parameterized type
    auto* input1 = graph->add_input();
    input1->set_name("input1");
    auto* input1_type = input1->mutable_type()->mutable_tensor_type();
    input1_type->set_elem_type(data_type);
    input1_type->mutable_shape()->add_dim()->set_dim_value(2);
    input1_type->mutable_shape()->add_dim()->set_dim_value(3);
    
    auto* input2 = graph->add_input();
    input2->set_name("input2");
    auto* input2_type = input2->mutable_type()->mutable_tensor_type();
    input2_type->set_elem_type(data_type);
    input2_type->mutable_shape()->add_dim()->set_dim_value(2);
    input2_type->mutable_shape()->add_dim()->set_dim_value(3);
    
    // Create output
    auto* output = graph->add_output();
    output->set_name("output");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(data_type);
    output_type->mutable_shape()->add_dim()->set_dim_value(2);
    output_type->mutable_shape()->add_dim()->set_dim_value(3);
    
    // Create Add node
    auto* node = graph->add_node();
    node->set_op_type("Add");
    node->add_input("input1");
    node->add_input("input2");
    node->add_output("output");
    
    model.mutable_graph()->clear_value_info();
    
    ASSERT_NO_THROW(InferTypes(model, false, false, false));
    ASSERT_NO_THROW(checker::check_model(model));
}

INSTANTIATE_TEST_SUITE_P(
    DifferentTypes,
    TypeInferenceParameterizedTest,
    ::testing::Values(