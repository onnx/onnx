#!/usr/bin/env python3

import os
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper, TensorProto, GraphProto, ModelProto, ValueInfoProto
from onnx.shape_inference import infer_types, infer_types_path
from onnx.test import test_util


class TestTypeInference(unittest.TestCase):
    """Test cases for type inference functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_simple_model(self, input_type=TensorProto.FLOAT, include_value_info=False):
        """Create a simple Add model for testing."""
        # Create input/output value info
        input1 = helper.make_tensor_value_info('input1', input_type, [2, 3])
        input2 = helper.make_tensor_value_info('input2', input_type, [2, 3])
        output = helper.make_tensor_value_info('output', input_type, [2, 3])
        
        # Create the Add node
        add_node = helper.make_node('Add', ['input1', 'input2'], ['output'])
        
        # Create graph
        inputs = [input1, input2]
        outputs = [output]
        value_info = [output] if include_value_info else []
        
        graph = helper.make_graph([add_node], 'test_graph', inputs, outputs, value_info)
        model = helper.make_model(graph)
        
        return model

    def _create_model_with_initializer(self):
        """Create a model with initializers for testing."""
        # Create input
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
        
        # Create initializer (weight)
        weight_data = np.random.randn(3, 4).astype(np.float32)
        weight_init = helper.make_tensor('weight', TensorProto.FLOAT, [3, 4], weight_data.flatten())
        
        # Create output
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
        
        # Create MatMul node
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['output'])
        
        # Create graph
        graph = helper.make_graph(
            [matmul_node], 
            'test_graph_with_init', 
            [input_tensor], 
            [output],
            initializer=[weight_init]
        )
        model = helper.make_model(graph)
        
        return model

    def _create_multi_node_model(self):
        """Create a model with multiple nodes for testing topological inference."""
        # Inputs
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3])
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 3])
        
        # Intermediate and final outputs
        intermediate = helper.make_tensor_value_info('intermediate', TensorProto.FLOAT, [2, 3])
        final_output = helper.make_tensor_value_info('final_output', TensorProto.FLOAT, [2, 3])
        
        # Nodes
        add_node = helper.make_node('Add', ['input1', 'input2'], ['intermediate'])
        relu_node = helper.make_node('Relu', ['intermediate'], ['final_output'])
        
        # Graph
        graph = helper.make_graph(
            [add_node, relu_node], 
            'multi_node_graph', 
            [input1, input2], 
            [final_output]
        )
        model = helper.make_model(graph)
        
        return model

    def test_infer_types_basic(self):
        """Test basic type inference functionality."""
        model = self._create_simple_model()
        
        # Remove value_info to test inference
        model.graph.ClearField('value_info')
        
        # Perform type inference
        inferred_model = infer_types(model)
        
        # Check that value_info has been populated
        self.assertGreater(len(inferred_model.graph.value_info), 0)
        
        # Check that output type is correctly inferred
        found_output_info = False
        for value_info in inferred_model.graph.value_info:
            if value_info.name == 'output':
                found_output_info = True
                self.assertEqual(value_info.type.tensor_type.elem_type, TensorProto.FLOAT)
        
        # Note: We might not always find output in value_info if it's already in graph.output
        # The key is that no errors occurred and the model is valid

    def test_infer_types_with_initializers(self):
        """Test type inference with initializers."""
        model = self._create_model_with_initializer()
        model.graph.ClearField('value_info')
        
        inferred_model = infer_types(model)
        
        # Verify the model is still valid
        onnx.checker.check_model(inferred_model)

    def test_infer_types_multi_node(self):
        """Test type inference with multiple nodes."""
        model = self._create_multi_node_model()
        model.graph.ClearField('value_info')
        
        inferred_model = infer_types(model)
        
        # Check that intermediate values have been inferred
        intermediate_found = False
        for value_info in inferred_model.graph.value_info:
            if value_info.name == 'intermediate':
                intermediate_found = True
                self.assertEqual(value_info.type.tensor_type.elem_type, TensorProto.FLOAT)
        
        self.assertTrue(intermediate_found, "Intermediate value type should be inferred")

    def test_infer_types_different_input_types(self):
        """Test type inference with different input types."""
        for input_type in [TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT32]:
            with self.subTest(input_type=input_type):
                model = self._create_simple_model(input_type)
                model.graph.ClearField('value_info')
                
                inferred_model = infer_types(model)
                
                # Verify model is valid
                onnx.checker.check_model(inferred_model)

    def test_infer_types_strict_mode(self):
        """Test strict mode behavior."""
        model = self._create_simple_model()
        
        # Test that strict_mode=True doesn't raise errors for valid models
        try:
            inferred_model = infer_types(model, strict_mode=True)
            onnx.checker.check_model(inferred_model)
        except Exception as e:
            self.fail(f"Strict mode raised unexpected error: {e}")

    def test_infer_types_check_type_flag(self):
        """Test the check_type flag."""
        model = self._create_simple_model()
        
        # Test with check_type=True
        try:
            inferred_model = infer_types(model, check_type=True)
            onnx.checker.check_model(inferred_model)
        except Exception as e:
            self.fail(f"check_type=True raised unexpected error: {e}")

    def test_infer_types_data_prop_flag(self):
        """Test the data_prop flag."""
        model = self._create_simple_model()
        
        # Test with data_prop=True
        try:
            inferred_model = infer_types(model, data_prop=True)
            onnx.checker.check_model(inferred_model)
        except Exception as e:
            self.fail(f"data_prop=True raised unexpected error: {e}")

    def test_infer_types_bytes_input(self):
        """Test type inference with bytes input."""
        model = self._create_simple_model()
        model_bytes = model.SerializeToString()
        
        inferred_model = infer_types(model_bytes)
        
        # Verify the result is a valid ModelProto
        self.assertIsInstance(inferred_model, ModelProto)
        onnx.checker.check_model(inferred_model)

    def test_infer_types_invalid_input_type(self):
        """Test that invalid input types raise appropriate errors."""
        with self.assertRaises(TypeError):
            infer_types("invalid_string_path")
        
        with self.assertRaises(TypeError):
            infer_types(123)

    def test_infer_types_path(self):
        """Test path-based type inference."""
        model = self._create_simple_model()
        model.graph.ClearField('value_info')
        
        # Save model to file
        model_path = os.path.join(self.temp_dir, 'test_model.onnx')
        onnx.save(model, model_path)
        
        # Test infer_types_path with separate output path
        output_path = os.path.join(self.temp_dir, 'inferred_model.onnx')
        infer_types_path(model_path, output_path)
        
        # Load and verify the inferred model
        inferred_model = onnx.load(output_path)
        onnx.checker.check_model(inferred_model)

    def test_infer_types_path_overwrite(self):
        """Test path-based type inference with overwrite."""
        model = self._create_simple_model()
        model.graph.ClearField('value_info')
        
        # Save model to file
        model_path = os.path.join(self.temp_dir, 'test_model.onnx')
        onnx.save(model, model_path)
        
        # Test infer_types_path without output path (should overwrite)
        infer_types_path(model_path)
        
        # Load and verify the inferred model
        inferred_model = onnx.load(model_path)
        onnx.checker.check_model(inferred_model)

    def test_infer_types_path_invalid_inputs(self):
        """Test that invalid inputs to infer_types_path raise appropriate errors."""
        # Test with ModelProto instead of path
        model = self._create_simple_model()
        with self.assertRaises(TypeError):
            infer_types_path(model)
        
        # Test with invalid path types
        with self.assertRaises(TypeError):
            infer_types_path(123)
        
        with self.assertRaises(TypeError):
            infer_types_path("valid_path", 123)  # Invalid output_path type

    def test_infer_types_path_pathlike(self):
        """Test path-based inference with PathLike objects."""
        from pathlib import Path
        
        model = self._create_simple_model()
        model.graph.ClearField('value_info')
        
        # Save model to file using Path object
        model_path = Path(self.temp_dir) / 'test_model.onnx'
        onnx.save(model, str(model_path))
        
        # Test with Path objects
        output_path = Path(self.temp_dir) / 'inferred_model.onnx'
        infer_types_path(model_path, output_path)
        
        # Verify the result
        inferred_model = onnx.load(str(output_path))
        onnx.checker.check_model(inferred_model)

    def test_infer_types_preserves_existing_info(self):
        """Test that existing value_info is preserved."""
        model = self._create_simple_model(include_value_info=True)
        original_value_info_count = len(model.graph.value_info)
        
        inferred_model = infer_types(model)
        
        # Should have at least the original value_info entries
        self.assertGreaterEqual(len(inferred_model.graph.value_info), original_value_info_count)

    def test_unknown_operator_handling(self):
        """Test handling of unknown operators."""
        # Create a model with a non-existent operator
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
        
        # Create a node with an unknown operator
        unknown_node = helper.make_node('UnknownOp', ['input'], ['output'])
        
        graph = helper.make_graph([unknown_node], 'unknown_op_graph', [input_tensor], [output_tensor])
        model = helper.make_model(graph)
        model.graph.ClearField('value_info')
        
        # Should not crash in non-strict mode
        inferred_model = infer_types(model, strict_mode=False)
        self.assertIsInstance(inferred_model, ModelProto)


class TestTypeInferenceEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_model(self):
        """Test type inference on an empty model."""
        graph = helper.make_graph([], 'empty_graph', [], [])
        model = helper.make_model(graph)
        
        # Should not crash
        inferred_model = infer_types(model)
        self.assertIsInstance(inferred_model, ModelProto)

    def test_model_with_no_inputs(self):
        """Test model with no inputs (only initializers)."""
        # Create initializer
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        init = helper.make_tensor('const', TensorProto.FLOAT, [2, 2], data)
        
        # Create output
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
        
        # Create Identity node
        node = helper.make_node('Identity', ['const'], ['output'])
        
        graph = helper.make_graph([node], 'no_input_graph', [], [output], initializer=[init])
        model = helper.make_model(graph)
        
        inferred_model = infer_types(model)
        onnx.checker.check_model(inferred_model)


if __name__ == '__main__':
    unittest.main()