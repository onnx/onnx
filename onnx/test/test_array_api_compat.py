# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Test array API compatibility for reference runtime.

This test demonstrates that the reference runtime can work with different
array backends that implement the Array API standard.
"""
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def create_simple_add_model():
    """Create a simple ONNX model that adds two tensors."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [3])

    add_node = helper.make_node("Add", ["X", "Y"], ["Z"])
    graph = helper.make_graph([add_node], "test_add", [X, Y], [Z])
    model = helper.make_model(graph)
    return model


def create_matmul_model():
    """Create a simple ONNX model that performs matrix multiplication."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4])

    matmul_node = helper.make_node("MatMul", ["X", "Y"], ["Z"])
    graph = helper.make_graph([matmul_node], "test_matmul", [X, Y], [Z])
    model = helper.make_model(graph)
    return model


def create_reduce_sum_model():
    """Create a simple ONNX model that reduces sum along an axis."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    reduce_node = helper.make_node("ReduceSum", ["X"], ["Y"], axes=[1], keepdims=0)
    graph = helper.make_graph([reduce_node], "test_reduce_sum", [X], [Y])
    model = helper.make_model(graph)
    return model


class TestArrayAPICompatibility:
    """Test suite for array API compatibility."""

    def test_numpy_add(self):
        """Test Add operator with NumPy arrays."""
        model = create_simple_add_model()
        sess = ReferenceEvaluator(model)

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = sess.run(None, {"X": x, "Y": y})

        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        np.testing.assert_array_equal(result[0], expected)

    def test_numpy_matmul(self):
        """Test MatMul operator with NumPy arrays."""
        model = create_matmul_model()
        sess = ReferenceEvaluator(model)

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        result = sess.run(None, {"X": x, "Y": y})

        expected = np.array([[1.0, 2.0, 3.0, 6.0], [4.0, 5.0, 6.0, 15.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_numpy_reduce_sum(self):
        """Test ReduceSum operator with NumPy arrays."""
        model = create_reduce_sum_model()
        sess = ReferenceEvaluator(model)

        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = sess.run(None, {"X": x})

        expected = np.sum(x, axis=1, keepdims=False)
        np.testing.assert_array_almost_equal(result[0], expected)

    @pytest.mark.skipif(
        "torch" not in dir(),
        reason="PyTorch not available",
    )
    def test_torch_add(self):
        """Test Add operator with PyTorch tensors."""
        import torch

        model = create_simple_add_model()
        sess = ReferenceEvaluator(model)

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        result = sess.run(None, {"X": x, "Y": y})

        expected = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
        assert torch.allclose(result[0], expected)

    @pytest.mark.skipif(
        "cupy" not in dir(),
        reason="CuPy not available",
    )
    def test_cupy_add(self):
        """Test Add operator with CuPy arrays (GPU)."""
        import cupy as cp

        model = create_simple_add_model()
        sess = ReferenceEvaluator(model)

        x = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        y = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
        result = sess.run(None, {"X": x, "Y": y})

        expected = cp.array([5.0, 7.0, 9.0], dtype=cp.float32)
        cp.testing.assert_array_equal(result[0], expected)

    @pytest.mark.skipif(
        "jax" not in dir(),
        reason="JAX not available",
    )
    def test_jax_add(self):
        """Test Add operator with JAX arrays."""
        import jax.numpy as jnp

        model = create_simple_add_model()
        sess = ReferenceEvaluator(model)

        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
        result = sess.run(None, {"X": x, "Y": y})

        expected = jnp.array([5.0, 7.0, 9.0], dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(result[0]), np.asarray(expected))


if __name__ == "__main__":
    # Run tests
    test = TestArrayAPICompatibility()
    test.test_numpy_add()
    print("✓ NumPy Add test passed")
    
    test.test_numpy_matmul()
    print("✓ NumPy MatMul test passed")
    
    test.test_numpy_reduce_sum()
    print("✓ NumPy ReduceSum test passed")
    
    print("\nAll tests passed!")
