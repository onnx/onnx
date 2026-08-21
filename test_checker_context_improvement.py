#!/usr/bin/env python3
# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""
Test case for ONNX checker context improvement.

This test verifies that ValidationError messages now include helpful context
about which specific input, output, initializer, or function has validation issues.
"""

import unittest
import tempfile
import os

import onnx
from onnx import checker, helper, TensorProto, ValueInfoProto, TypeProto, GraphProto, ModelProto


class TestCheckerContextImprovement(unittest.TestCase):
    """Test cases to verify improved error messages in ONNX checker."""

    def test_input_validation_error_includes_context(self):
        """Test that validation errors for graph inputs include the input name."""
        # Create a ValueInfoProto with missing shape field
        input_value_info = ValueInfoProto()
        input_value_info.name = "problematic_input"
        
        # Create a type with tensor_type but intentionally missing shape
        type_proto = TypeProto()
        tensor_type = type_proto.tensor_type
        tensor_type.elem_type = TensorProto.FLOAT
        # Note: intentionally NOT setting shape to trigger the error
        
        input_value_info.type.CopyFrom(type_proto)
        
        # Create a minimal graph with this problematic input
        graph = GraphProto()
        graph.name = "test_graph"
        graph.input.extend([input_value_info])
        
        # Create a minimal output to make the graph valid otherwise
        output_value_info = helper.make_value_info("output", TensorProto.FLOAT, [1])
        graph.output.extend([output_value_info])
        
        # Create a minimal node to make the graph valid otherwise
        node = helper.make_node("Identity", ["problematic_input"], ["output"])
        graph.node.extend([node])
        
        # Test that the error message includes the input name
        with self.assertRaises(checker.ValidationError) as cm:
            checker.check_graph(graph)
        
        error_message = str(cm.exception)
        self.assertIn("problematic_input", error_message, 
                      "Error message should include the name of the problematic input")
        self.assertIn("Bad input specification", error_message,
                      "Error message should indicate this is an input specification problem")

    def test_output_validation_error_includes_context(self):
        """Test that validation errors for graph outputs include the output name."""
        # Create a ValueInfoProto with missing shape field
        output_value_info = ValueInfoProto()
        output_value_info.name = "problematic_output"
        
        # Create a type with tensor_type but intentionally missing shape
        type_proto = TypeProto()
        tensor_type = type_proto.tensor_type
        tensor_type.elem_type = TensorProto.FLOAT
        # Note: intentionally NOT setting shape to trigger the error
        
        output_value_info.type.CopyFrom(type_proto)
        
        # Create a minimal graph with this problematic output
        graph = GraphProto()
        graph.name = "test_graph"
        
        # Create a valid input
        input_value_info = helper.make_value_info("input", TensorProto.FLOAT, [1])
        graph.input.extend([input_value_info])
        
        # Add the problematic output
        graph.output.extend([output_value_info])
        
        # Create a minimal node
        node = helper.make_node("Identity", ["input"], ["problematic_output"])
        graph.node.extend([node])
        
        # Test that the error message includes the output name
        with self.assertRaises(checker.ValidationError) as cm:
            checker.check_graph(graph)
        
        error_message = str(cm.exception)
        self.assertIn("problematic_output", error_message,
                      "Error message should include the name of the problematic output")
        self.assertIn("Bad output specification", error_message,
                      "Error message should indicate this is an output specification problem")

    def test_initializer_validation_error_includes_context(self):
        """Test that validation errors for initializers include the tensor name."""
        # Create a tensor with missing data_type field
        problematic_tensor = TensorProto()
        problematic_tensor.name = "problematic_initializer"
        # Note: intentionally NOT setting data_type to trigger the error
        problematic_tensor.dims.extend([2, 3])
        problematic_tensor.float_data.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Create a minimal graph with this problematic initializer
        graph = GraphProto()
        graph.name = "test_graph"
        
        # Add valid input and output
        input_value_info = helper.make_value_info("input", TensorProto.FLOAT, [2, 3])
        output_value_info = helper.make_value_info("output", TensorProto.FLOAT, [2, 3])
        graph.input.extend([input_value_info])
        graph.output.extend([output_value_info])
        
        # Add the problematic initializer
        graph.initializer.extend([problematic_tensor])
        
        # Create a minimal node
        node = helper.make_node("Add", ["input", "problematic_initializer"], ["output"])
        graph.node.extend([node])
        
        # Test that the error message includes the initializer name
        with self.assertRaises(checker.ValidationError) as cm:
            checker.check_graph(graph)
        
        error_message = str(cm.exception)
        self.assertIn("problematic_initializer", error_message,
                      "Error message should include the name of the problematic initializer")
        self.assertIn("Bad initializer specification", error_message,
                      "Error message should indicate this is an initializer specification problem")


if __name__ == "__main__":
    unittest.main()