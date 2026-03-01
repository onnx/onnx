---
name: update-onnx-operator
description: Guide for updating an existing ONNX operator to a new version when making breaking changes to its signature, behavior, or supported types.
license: Apache-2.0
---

# Update ONNX Operator

This skill helps you update an existing ONNX operator to a new version when making breaking changes.

## When to Use This Skill

Use this skill when you need to:
- Make breaking changes to an operator's signature (inputs, outputs, attributes)
- Change an operator's behavior or semantics
- Add or remove supported types for an operator
- Update an operator to support new functionality

## Checklist for Updating an Operator

When updating an operator to a new version, follow these steps:

### 1. Increment Domain Version
- **File:** `onnx/defs/schema.h`
- **Location:** `DomainToVersionRange::DomainToVersionRange()` constructor
- **Action:** Increment the max version for the domain (e.g., ONNX_DOMAIN from 26 to 27)

### 2. Preserve Old Schema
- **File:** `onnx/defs/<category>/old.cc`
- **Action:** Copy the complete old operator schema from `defs.cc` to `old.cc`
- **Note:** Keep old schema permanently for version history

### 3. Update Operator Schema
- **File:** `onnx/defs/<category>/defs.cc`
- **Action:** 
  - Update `SinceVersion()` to the new version number
  - Modify inputs, outputs, attributes as needed
  - Update documentation
  - Update type constraints
  - Update shape inference if needed

### 4. Register New Version
- **File:** `onnx/defs/operator_sets.h`
- **Action:**
  - Add forward declaration for new operator version
  - Add to appropriate OpSet class
  - Ensure registration in `RegisterOnnxOperatorSetSchema()`

### 5. Add Version Adapter
- **Files:** 
  - `onnx/version_converter/adapters/<operator>_<old>_<new>.h` (if complex)
  - `onnx/version_converter/convert.h`
- **Action:**
  - Use `CompatibleAdapter` for simple changes
  - Create custom adapter for complex changes
  - Register adapter in `DefaultVersionConverter` constructor
  - Optional: Add downgrade adapter

### 6. Update Tests
- **Files:**
  - `onnx/backend/test/case/node/<operator>.py`
  - `onnx/test/version_converter/automatic_upgrade_test.py`
  - `onnx/test/version_converter/automatic_downgrade_test.py`
  - `onnx/test/shape_inference_test.py` (if applicable)
- **Action:**
  - Add backend tests for new functionality
  - Add automatic upgrade test with `_test_op_upgrade`
  - Add automatic downgrade test with `_test_op_downgrade`
  - Update shape inference tests if needed

### 7. Generate Documentation
- **Command:** `bash tools/update_doc.sh`
- **Action:**
  - Generates updated `docs/Operators.md`
  - Creates test data in `onnx/backend/test/data/node/`
  - Clean up with `python onnx/backend/test/cmd_tools.py generate-data --clean` if needed

## Example Usage

```bash
# After making all code changes, generate documentation
bash tools/update_doc.sh

# Clean up test data if needed
python onnx/backend/test/cmd_tools.py generate-data --clean

# Run tests
python -m pytest onnx/test/version_converter/automatic_upgrade_test.py::test_<operator>_upgrade
```

## Breaking vs Non-Breaking Changes

**Breaking Changes** (require new version):
- Adding/removing/renaming attributes (even optional ones)
- Adding/removing/reordering inputs or outputs
- Adding/removing supported types
- Changing operator behavior

**Non-Breaking Changes** (no new version needed):
- Clarifying ambiguous specifications to match implementation

## Resources

- [Full Updating Operator Guide](../../docs/UpdatingOperator.md)
- [ONNX Versioning](../../docs/Versioning.md)
- [Adding New Operators](../../docs/AddNewOp.md)
- [Version Converter](../../docs/VersionConverter.md)

## Quick Reference

| Step | File | Action |
|------|------|--------|
| 1 | `onnx/defs/schema.h` | Increment domain version |
| 2 | `onnx/defs/<cat>/old.cc` | Copy old schema |
| 3 | `onnx/defs/<cat>/defs.cc` | Update schema + SinceVersion |
| 4 | `onnx/defs/operator_sets.h` | Register new version |
| 5 | `onnx/version_converter/convert.h` | Add version adapter |
| 6 | `onnx/backend/test/case/node/` | Add tests |
| 7 | Run `tools/update_doc.sh` | Generate docs |
