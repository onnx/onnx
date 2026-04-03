<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# External Data Security

This document describes the security model for loading and saving external data files in ONNX models. It is intended for maintainers working on the external data code paths.

## Threat Model

When an ONNX model references external data files via relative paths, an attacker who controls the model file can attempt:

- **Symlink traversal**: A final-component symlink in the external data path pointing to a sensitive file (e.g., `/etc/shadow`), causing ONNX to read or overwrite arbitrary files.
- **Parent-directory symlink**: A symlink in a parent directory component of the external data path, bypassing a check that only inspects the final component.
- **Hardlink attacks**: A hardlink to a sensitive file appearing as a normal file, bypassing symlink-only checks while still exposing unintended data.
- **Path traversal**: Using `..` segments or absolute paths to escape the model directory.

## Defense Layers

We use a 4-layer defense-in-depth approach. Each entry point applies the layers appropriate to its context (see the table below).

### Layer 1: Canonical Path Containment

- **C++**: `std::filesystem::weakly_canonical()` resolves the path, then verifies it starts with the canonical base directory.

This catches `..` traversal and symlinks in any path component (not just the final one).

### Layer 2: Symlink Detection

- **C++**: `std::filesystem::is_symlink(data_path)` rejects the final-component symlink.

This is a belt-and-suspenders check alongside containment. It provides a clear, specific error message when the final path component is a symlink.

### Layer 3: Secure File Open with Post-Open Verification

Three platform-specific strategies, in order of preference:

- **Linux 5.6+**: `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` — atomic kernel-level containment and symlink rejection. No TOCTOU possible. Falls back at runtime if the kernel is too old (`ENOSYS`).
- **FreeBSD 13+ / macOS 15+**: `openat(O_RESOLVE_BENEATH | O_NOFOLLOW)` — equivalent kernel-level containment via BSD-style flags.
- **POSIX fallback**: `open(O_NOFOLLOW)` followed by inode comparison (`stat` canonical path vs `fstat` fd) to verify the opened fd matches the validated path. This narrows the TOCTOU window to a race between `open()` and `stat()`.
- **Windows**: `CreateFileW(FILE_FLAG_OPEN_REPARSE_POINT)` opens without following symlinks/junctions, then checks `dwFileAttributes` for reparse points and verifies via inode comparison (`dwVolumeSerialNumber` + `nFileIndexHigh/Low` from `GetFileInformationByHandle()`). Returns a raw OS HANDLE (not a CRT fd) to avoid fd table mismatch across DLL boundaries; Python converts via `msvcrt.open_osfhandle()`.

Hardlinks are detected via `fstat()` link count (POSIX) or `GetFileInformationByHandle()` (Windows), always fail closed.

### Layer 4: Pre-Open Hardlink Count Check

- **C++**: `std::filesystem::hard_link_count(data_path) > 1` rejects files with multiple hardlinks before opening.

This is defense-in-depth alongside Layer 3's post-open hardlink check. It provides a clear error message at the path level before any file I/O occurs.

## Protected Entry Points

All Python entry points that open external data files use the C++ `open_external_data()` function, which applies Layers 1-4 in a single call. The C++ checker (`resolve_external_data_location`) validates paths without opening files, so Layer 3 does not apply there.

| Entry Point | File | Layers |
|---|---|---|
| `resolve_external_data_location` | `onnx/checker.cc` | 1, 2, 4 |
| `open_external_data` | `onnx/checker.cc` | 1, 2, 3, 4 |
| `load_external_data_for_tensor` | `onnx/external_data_helper.py` | 1, 2, 3, 4 (via `open_external_data`) |
| `save_external_data` | `onnx/external_data_helper.py` | 1, 3 (via `open_external_data`) |
| `ModelContainer._load_large_initializers` | `onnx/model_container.py` | 1, 2, 3, 4 (via `open_external_data`) |

## Known Limitations

### Windows

- Reparse points (symlinks, junctions) are rejected via `FILE_FLAG_OPEN_REPARSE_POINT` + attribute check.
- Symlink and hardlink tests are skipped on Windows in the test suite.

### Case-Insensitive Filesystems

The canonical path containment checks (Layers 1 and 3) use string comparison. On case-insensitive filesystems (Windows NTFS, macOS HFS+), paths with different casing may incorrectly fail containment. This fails closed (false rejection, not a bypass).

## Testing

Test coverage is in:

- **C++**: `onnx/test/cpp/checker_test.cc` — `*SymLink*` tests for symlink detection and containment, `OpenExternalData*` for secure open and verification.
- **Python**: `onnx/test/test_external_data.py`:
  - `TestSaveExternalDataSymlinkProtection` — save-side symlink rejection.
  - `TestLoadExternalDataSymlinkProtection` — load-side symlink rejection, parent-directory symlink, `load_external_data_for_model` rejection.
  - `TestLoadExternalDataHardlinkProtection` — load-side hardlink rejection.
  - `TestSaveExternalDataAbsolutePathValidation` — absolute path rejection.

Symlink and hardlink tests are skipped on Windows (`os.name == "nt"`).

---

## External Data Attribute Validation

This section describes the security model for validating external data attributes in `ExternalDataInfo`. It covers defenses against attribute injection (CWE-915) and resource exhaustion (CWE-400) via crafted `external_data` entries in `TensorProto`.

**Advisory:** [GHSA-538c-55jv-c5g9](https://github.com/onnx/onnx/security/advisories/GHSA-538c-55jv-c5g9)

### Threat Model

An attacker provides a malicious ONNX model with crafted `external_data` entries in `TensorProto`. The `external_data` field is a repeated `StringStringEntryProto` — a key-value store that accepts arbitrary strings for both key and value.

The attack is triggered during `onnx.load()` with no explicit checker invocation required. `ExternalDataInfo.__init__` processes these key-value pairs to populate object attributes.

Attack vectors:

- **Arbitrary attribute injection**: Setting unknown keys (e.g. `evil_attr`) causes `setattr()` to create arbitrary attributes on the `ExternalDataInfo` object. While no current consumer iterates over attributes, injected attributes create latent risk for future code.
- **Dunder attribute injection**: Setting keys like `__class__` or `__dict__` corrupts the Python object's internal state, enabling type confusion attacks.
- **Negative offset/length**: Negative values for `offset` cause `file.seek()` to raise `OSError`. Negative `length` causes `file.read(-1)` to read the entire file to EOF, bypassing intended size limits.
- **Resource exhaustion (DoS)**: Setting `length` to a multi-petabyte value causes unbounded memory allocation when reading external data, even if the actual data file is small.

Four Python consumers of `ExternalDataInfo` exist: `load_external_data_for_tensor`, `set_external_data` / `write_external_data_tensors`, `ModelContainer._load_large_initializers`, and `ReferenceEvaluator` (in `onnx/reference/reference_evaluator.py`). (The C++ checker validates paths but does not use the Python `ExternalDataInfo` class.)

## Defense Layers

We use a 3-layer defense-in-depth approach. Each layer addresses a different class of attack and operates at a different point in the processing pipeline.

### Layer 1: Attribute Whitelist (CWE-915 Mitigation)

`ExternalDataInfo.__init__` only accepts keys in `_ALLOWED_EXTERNAL_DATA_KEYS`: `location`, `offset`, `length`, `checksum`, `basepath`. Unknown keys are warned via `warnings.warn()` and ignored — this prevents arbitrary attribute injection.

This also blocks dunder attribute injection (e.g. `__class__`, `__dict__`) that could cause object type confusion.

**Rationale**: While we cannot prevent someone from constructing malicious protobuf directly, rejecting unknown keys at the Python object level is defense-in-depth that limits the attack surface. The whitelist is a `frozenset` to prevent runtime mutation.

### Layer 2: Bounds Validation at Parse Time (CWE-400 Mitigation)

`offset` and `length` must be non-negative integers. Non-numeric strings raise `ValueError`. This catches obviously invalid values early, before any file I/O occurs.

**Rationale**: Negative `offset` causes `file.seek(-1)` to raise `OSError`; negative `length` causes `file.read(-1)` to read the entire file, bypassing intended size limits. Validating at parse time provides a clear error message at the point closest to the malicious input.

### Layer 3: File-Size Validation at Consumption Time (CWE-400 Mitigation, Defense-in-Depth)

In `load_external_data_for_tensor()` and `ModelContainer._load_large_initializers`, before reading: `offset <= file_size` and `offset + length <= file_size` are verified. A 1KB data file cannot cause a multi-petabyte memory allocation.

**Rationale**: This is the critical safety net. It prevents memory exhaustion regardless of how the model was constructed — even via direct protobuf APIs that bypass Python-level parsing entirely. Validation happens at the point of actual file I/O, the last opportunity before harm occurs.

## Why Layered Defense

- **Layer 1 (whitelist)** catches the broadest class of attacks at parse time. It blocks attribute injection, dunder corruption, and any future unknown-key attack vector.
- **Layer 2 (bounds validation)** catches obviously invalid numeric values at parse time, providing clear error messages.
- **Layer 3 (file-size validation)** is the critical safety net that prevents actual harm at the I/O boundary. This layer cannot be bypassed even if an attacker crafts a model using protobuf APIs directly, because validation happens at the point of actual file read.

## Protected Entry Points

| Entry Point | File | Layers |
|---|---|---|
| `ExternalDataInfo.__init__` | `onnx/external_data_helper.py` | 1, 2 |
| `load_external_data_for_tensor` | `onnx/external_data_helper.py` | 1, 2, 3 |
| `set_external_data` | `onnx/external_data_helper.py` | 1 (whitelist by overwrite) |
| `ModelContainer._load_large_initializers` | `onnx/model_container.py` | 1, 2, 3 |

## Testing

Test coverage is in `onnx/test/test_external_data.py`:

- `TestExternalDataInfoSecurity`:
  - **CWE-915 (attribute injection):** `test_unknown_key_rejected`, `test_dunder_key_rejected`, `test_multiple_unknown_keys_all_rejected`, `test_allowed_keys_constant_is_frozen`
  - **CWE-400 (bounds/DoS):** `test_negative_offset_rejected`, `test_negative_length_rejected`, `test_non_numeric_offset_raises`, `test_non_numeric_length_raises`
  - **Regression guards:** `test_valid_external_data_accepted`, `test_zero_offset_and_length_accepted`
- `TestLoadExternalDataFileSizeValidation`:
  - **File-size validation:** `test_offset_exceeds_file_size_raises`, `test_length_exceeds_available_data_raises`, `test_valid_offset_and_length_load_correctly`
