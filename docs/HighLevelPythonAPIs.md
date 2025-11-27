# High-Level Python APIs for ONNX

While the core `onnx` Python package provides low-level access to ONNX graphs and protobuf structures, several community packages offer higher-level, more user-friendly APIs for creating and manipulating ONNX models.

This document lists selected packages that provide high-level Python APIs for working with ONNX models.

## Official ONNX Project

### ir-py

[ir-py](https://github.com/onnx/ir-py) is the official intermediate representation library for ONNX in Python. It provides a Pythonic, user-friendly API for creating and manipulating ONNX graphs.

**Installation:**
```bash
pip install onnx-ir
```

**Key Features:**
- Pythonic API for building ONNX models
- Easy graph manipulation and transformation
- Direct integration with the ONNX specification
- Type-safe operations

## Community Packages

### ndonnx

[ndonnx](https://github.com/Quantco/ndonnx) provides a NumPy-like interface for building ONNX graphs. It allows users to write array operations that compile to ONNX models.

**Installation:**
```bash
pip install ndonnx
```

**Key Features:**
- NumPy-compatible API
- Automatic graph construction from array operations
- Support for eager and lazy evaluation
- Integration with the Array API standard

### spox

[spox](https://github.com/Quantco/spox) is a Python framework for constructing ONNX computational graphs. It provides a type-safe, functional approach to building ONNX models.

**Installation:**
```bash
pip install spox
```

**Key Features:**
- Type-safe graph construction
- Functional programming paradigm
- Composable operators
- Automatic type inference

## Comparison

| Package | Approach | Best For |
|---------|----------|----------|
| ir-py | Direct graph manipulation | Low-level graph editing, official tooling |
| ndonnx | NumPy-like API | Users familiar with NumPy, array computations |
| spox | Functional/type-safe | Type-safe model construction, composable operations |

## When to Use Each

- **ir-py**: Use when you need direct control over the ONNX graph structure or when working with official ONNX tooling.
- **ndonnx**: Use when you want to express computations in a NumPy-like style and have them automatically converted to ONNX.
- **spox**: Use when you want type safety and a functional programming approach to building ONNX models.

## Additional Resources

- [ONNX Python API Documentation](https://onnx.ai/onnx/api/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [ONNX Tutorials](https://github.com/onnx/tutorials)
