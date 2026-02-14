<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Skill: Updating an Operator to a New Version

This document provides a comprehensive checklist for updating an existing operator to a new version in ONNX. Use this guide when making breaking changes to an operator's signature or semantics.

## Table of Contents

- [Overview](#overview)
- [When to Update an Operator Version](#when-to-update-an-operator-version)
- [Complete Checklist](#complete-checklist)
  - [1. Update Domain Version Range](#1-update-domain-version-range)
  - [2. Move Old Operator Schema to old.cc](#2-move-old-operator-schema-to-oldcc)
  - [3. Update Operator Schema](#3-update-operator-schema)
  - [4. Register New Operator Version](#4-register-new-operator-version)
  - [5. Add Version Adapter](#5-add-version-adapter)
  - [6. Update Tests](#6-update-tests)
  - [7. Update Documentation](#7-update-documentation)
- [Example Workflow](#example-workflow)
- [Additional Resources](#additional-resources)

## Overview

When you need to make breaking changes to an operator (changes to inputs/outputs, attributes, behavior, or types), you must create a new version of that operator. This process involves updating multiple files across the ONNX codebase to ensure proper versioning, backward compatibility, and testing.

## When to Update an Operator Version

You MUST create a new operator version when making **breaking changes**, which include:

- Adding, removing, or renaming an attribute (even optional attributes with default values)
- Adding, removing, or reordering inputs or outputs
- Adding or removing supported types for inputs, outputs, or attributes
- Changing the operator's behavior or semantics (e.g., adding broadcasting support)

**Non-breaking changes** (do not require a new version):

- Clarifications of specification ambiguities to match prevailing implementation practice

## Complete Checklist

### 1. Update Domain Version Range

**File:** `onnx/defs/schema.h`

**Location:** `DomainToVersionRange::DomainToVersionRange()` constructor

**Action:** Increment the maximum version for the relevant domain.

```cpp
// Example: Incrementing ONNX_DOMAIN from 26 to 27
map_[ONNX_DOMAIN] = std::make_pair(1, 27);  // Changed from 26 to 27
```

**Domains:**
- `ONNX_DOMAIN` - Main ONNX operators (e.g., "")
- `AI_ONNX_ML_DOMAIN` - Machine learning operators
- `AI_ONNX_TRAINING_DOMAIN` - Training operators
- `AI_ONNX_PREVIEW_TRAINING_DOMAIN` - Preview training operators

### 2. Move Old Operator Schema to old.cc

**File:** `onnx/defs/<category>/old.cc` (e.g., `onnx/defs/math/old.cc`, `onnx/defs/logical/old.cc`)

**Action:** Copy the complete old operator schema definition from `defs.cc` to `old.cc` to preserve the old version.

**Example:**
```cpp
// In old.cc - preserve the old version
ONNX_OPERATOR_SET_SCHEMA(
    Add,
    12,  // old version number
    OpSchema()
        .SetDoc(...)
        .Input(...)
        .Output(...)
        // ... rest of old schema
);
```

**Note:** Keep the old schema exactly as it was. Do NOT remove it from the repository even after the new version is added.

### 3. Update Operator Schema

**File:** `onnx/defs/<category>/defs.cc` (e.g., `onnx/defs/math/defs.cc`)

**Action:** Update the existing operator schema in `defs.cc`:

1. Update `SinceVersion()` to the new opset version (from step 1)
2. Modify inputs, outputs, attributes, or behavior as needed
3. Update documentation to reflect changes
4. Update type constraints if needed
5. Update shape inference function if needed

**Example:**
```cpp
// In defs.cc - updated version
ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,  // new version number (incremented from step 1)
    OpSchema()
        .SetDoc("Updated documentation...")
        .SinceVersion(14)  // IMPORTANT: Update this!
        .Input(...)
        .Output(...)
        // ... new schema with changes
);
```

### 4. Register New Operator Version

**File:** `onnx/defs/operator_sets.h`

**Action:** Add forward declaration and registration for the new operator version.

**Steps:**

1. **Add forward declaration** for the new version at the top of the file:
   ```cpp
   // Example for version 14 of Add operator
   class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add);
   ```

2. **Add to OpSet class** for the new version:
   ```cpp
   // In the class definition for OpSet_Onnx_ver14
   class OpSet_Onnx_ver14 {
    public:
     static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
       // ... other operators ...
       fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add)>());
       // ... more operators ...
     }
   };
   ```

3. **Ensure registration** in `RegisterOnnxOperatorSetSchema()` function:
   ```cpp
   ONNX_API inline void RegisterOnnxOperatorSetSchema() {
     // ... existing registrations ...
     RegisterOpSetSchema<OpSet_Onnx_ver14>();
     // ... more registrations ...
   }
   ```

### 5. Add Version Adapter

**Files:**
- `onnx/version_converter/adapters/<operator_name>_<old>_<new>.h` (create new file)
- `onnx/version_converter/convert.h` (register adapter)

**Action:** Create a version adapter to enable conversion between operator versions.

**Steps:**

1. **Create adapter file** (if complex changes) or use `CompatibleAdapter`:

   **For compatible changes** (old schema valid under new schema):
   ```cpp
   // In convert.h
   registerAdapter(std::make_unique<CompatibleAdapter>("Add", OpSetID(13), OpSetID(14)));
   ```

   **For complex changes** (create new adapter file):
   ```cpp
   // In onnx/version_converter/adapters/add_13_14.h
   #pragma once
   #include "onnx/version_converter/adapters/adapter.h"
   
   namespace ONNX_NAMESPACE {
   namespace version_conversion {
   
   class Add_13_14 : public Adapter {
    public:
     explicit Add_13_14() : Adapter("Add", OpSetID(13), OpSetID(14)) {}
     
     Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
       // Implement conversion logic here
       // Modify node in place or create new node
       return node;
     }
   };
   
   } // namespace version_conversion
   } // namespace ONNX_NAMESPACE
   ```

2. **Register adapter in `convert.h`**:
   ```cpp
   // Include the header
   #include "onnx/version_converter/adapters/add_13_14.h"
   
   // In DefaultVersionConverter constructor
   DefaultVersionConverter() {
     // ... existing adapters ...
     
     /******** 13 -> 14 ********/
     registerAdapter(std::make_unique<Add_13_14>());
     
     // ... more adapters ...
   }
   ```

3. **(Optional) Add downgrade adapter** for converting from new to old version:
   ```cpp
   /******** 14 -> 13 ********/
   registerAdapter(std::make_unique<Add_14_13>());
   ```

### 6. Update Tests

**Files:**
- `onnx/backend/test/case/node/<operator_name>.py` - Backend test cases
- `onnx/test/version_converter/automatic_upgrade_test.py` - Automatic upgrade tests
- `onnx/test/version_converter/automatic_downgrade_test.py` - Automatic downgrade tests
- `onnx/test/shape_inference_test.py` - Shape inference tests (if applicable)

**Action:** Add or update tests for the new operator version.

**Steps:**

1. **Update backend tests** in `onnx/backend/test/case/node/<operator>.py`:
   - Add test cases that exercise new functionality
   - Ensure tests cover main usage and corner cases
   - Tests will be used to generate documentation examples

   ```python
   # Example in onnx/backend/test/case/node/add.py
   @staticmethod
   def export_add_broadcast() -> None:
       node = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])
       x = np.random.randn(3, 4, 5).astype(np.float32)
       y = np.random.randn(5).astype(np.float32)
       z = x + y
       expect(node, inputs=[x, y], outputs=[z], name="test_add_broadcast")
   ```

2. **Add automatic upgrade test** in `automatic_upgrade_test.py`:
   ```python
   def test_add_upgrade(self) -> None:
       self._test_op_upgrade("Add", 13, 14)  # from version 13 to 14
   ```

3. **Add automatic downgrade test** in `automatic_downgrade_test.py`:
   ```python
   def test_add_downgrade(self) -> None:
       self._test_op_downgrade("Add", 14, 13)  # from version 14 to 13
   ```

4. **Update shape inference tests** in `shape_inference_test.py` (if shape inference changed):
   ```python
   def test_add_shape_inference(self) -> None:
       # Add tests for shape inference with new operator version
       pass
   ```

### 7. Update Documentation

**Action:** Generate updated documentation and test data.

**Steps:**

1. **Run documentation update script**:
   ```bash
   bash tools/update_doc.sh
   ```
   
   This script will:
   - Update `docs/Operators.md` with new operator documentation
   - Generate test data files in `onnx/backend/test/data/node/`

2. **Clean up test data** (if needed):
   ```bash
   python onnx/backend/test/cmd_tools.py generate-data --clean
   ```
   
   This ensures only necessary test data is preserved.

3. **Verify generated documentation**:
   - Check `docs/Operators.md` for the new operator version
   - Ensure the documentation clearly describes the changes
   - Verify that examples are included and accurate

## Example Workflow

Here's a complete example of updating the `Add` operator from version 13 to version 14 to add support for a new data type:

### Step-by-step Example

1. **Increment domain version** in `onnx/defs/schema.h`:
   ```cpp
   map_[ONNX_DOMAIN] = std::make_pair(1, 14);  // Changed from 13 to 14
   ```

2. **Copy old schema** to `onnx/defs/math/old.cc`:
   ```cpp
   // Preserve version 13 of Add
   ONNX_OPERATOR_SET_SCHEMA(
       Add,
       13,
       OpSchema()
           .SetDoc("Add version 13 documentation")
           .SinceVersion(13)
           // ... complete old schema ...
   );
   ```

3. **Update schema** in `onnx/defs/math/defs.cc`:
   ```cpp
   ONNX_OPERATOR_SET_SCHEMA(
       Add,
       14,
       OpSchema()
           .SetDoc("Add version 14 - now supports bfloat16")
           .SinceVersion(14)  // Updated version
           .Input(0, "A", "First operand", "T")
           .Input(1, "B", "Second operand", "T")
           .Output(0, "C", "Result", "T")
           .TypeConstraint("T", 
               {"tensor(float)", "tensor(int32)", "tensor(bfloat16)"},  // Added bfloat16
               "Input and output types")
           // ... rest of schema ...
   );
   ```

4. **Register in operator_sets.h**:
   ```cpp
   // Forward declaration
   class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add);
   
   // In OpSet_Onnx_ver14 class
   fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Add)>());
   ```

5. **Add version adapter** in `convert.h`:
   ```cpp
   // Since old schema is compatible with new one, use CompatibleAdapter
   registerAdapter(std::make_unique<CompatibleAdapter>("Add", OpSetID(13), OpSetID(14)));
   ```

6. **Add tests**:
   - Update `onnx/backend/test/case/node/add.py` with bfloat16 tests
   - Add upgrade test in `automatic_upgrade_test.py`
   - Add downgrade test in `automatic_downgrade_test.py`

7. **Generate documentation**:
   ```bash
   bash tools/update_doc.sh
   ```

## Additional Resources

- [ONNX Versioning Documentation](Versioning.md) - Detailed versioning policies
- [Adding New Operator Documentation](AddNewOp.md) - Guide for adding new operators
- [Version Converter Documentation](VersionConverter.md) - Information about version adapters
- [Operator Conventions](OpConventions.md) - Best practices for operator design

## Summary

When updating an operator version, remember to:

✅ Increment the domain version in `schema.h`  
✅ Preserve old schema in `old.cc`  
✅ Update schema in `defs.cc` with new `SinceVersion()`  
✅ Register new version in `operator_sets.h`  
✅ Add version adapter in `convert.h`  
✅ Add upgrade/downgrade tests  
✅ Update backend tests  
✅ Generate documentation with `tools/update_doc.sh`  

Following this checklist ensures proper versioning, maintains backward compatibility, and keeps the ONNX specification consistent and well-tested.
