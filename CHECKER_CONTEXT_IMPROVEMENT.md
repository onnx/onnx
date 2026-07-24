# ONNX Checker Context Improvement

## Problem
When the ONNX checker throws ValidationError about missing fields like:
```
Field 'shape' of 'type' is required but missing.
```

It doesn't provide information about which specific object in the model is incomplete, making it difficult for users to locate and fix the error.

## Solution
Enhanced the ONNX checker to add contextual information using the existing `ValidationError::AppendContext` mechanism. The checker now provides specific information about which component has the validation issue.

## Changes Made

### 1. Graph Input Validation
- **Location**: `check_graph()` function, input checking loop
- **Enhancement**: Added try-catch wrapper around `check_value_info()` calls for graph inputs
- **Context Message**: `"Bad input specification for input. Name: {input_name}"`

### 2. Graph Output Validation
- **Location**: `check_graph()` function, output checking loop  
- **Enhancement**: Added try-catch wrapper around `check_value_info()` calls for graph outputs
- **Context Message**: `"Bad output specification for output. Name: {output_name}"`

### 3. Initializer Validation
- **Location**: `check_graph()` function, initializer checking loop
- **Enhancement**: Added try-catch wrapper around `check_tensor()` calls for initializers
- **Context Message**: `"Bad initializer specification for tensor. Name: {tensor_name}"`

### 4. Sparse Initializer Validation
- **Location**: `check_graph()` function, sparse initializer checking loop
- **Enhancement**: Added try-catch wrapper around `check_sparse_tensor()` calls
- **Context Message**: `"Bad sparse initializer specification for tensor. Name: {tensor_name}"`

### 5. Function Validation
- **Location**: `check_model_local_functions()` function
- **Enhancement**: Added try-catch wrapper around `check_function()` calls
- **Context Message**: `"Bad function specification for function. Name: {function_name}"`

## Example Improvement

**Before:**
```
ValidationError: Field 'shape' of 'type' is required but missing.
```

**After:**
```
ValidationError: Field 'shape' of 'type' is required but missing.

==> Context: Bad input specification for input. Name: my_problematic_input
```

## Implementation Pattern
The changes follow the existing pattern used elsewhere in the checker:

```cpp
ONNX_TRY {
  check_value_info(value_info, ctx);
}
ONNX_CATCH(ValidationError & ex) {
  ONNX_HANDLE_EXCEPTION([&]() {
    ex.AppendContext("Bad input specification for input. Name: " + value_info.name());
    ONNX_THROW_EX(ex);
  });
}
```

## Impact
- **User Experience**: Dramatically improves debugging experience by making it clear which specific component has validation issues
- **Compatibility**: No breaking changes - error types and core messages remain the same
- **Performance**: Minimal overhead - try-catch only activates when errors occur

## Testing
A test suite has been provided (`test_checker_context_improvement.py`) that verifies the enhanced error messages include the expected context information for various validation scenarios.

The changes affect the core validation paths that users encounter when their ONNX models have specification errors, making this improvement highly valuable for the developer experience.