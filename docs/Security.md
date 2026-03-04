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

We use a 4-layer defense-in-depth approach. Each layer is applied at every entry point that opens external data files.

### Layer 1: Canonical Path Containment

- **C++**: `std::filesystem::weakly_canonical()` resolves the path, then verifies it starts with the canonical base directory.
- **Python**: `os.path.realpath()` resolves all symlinks in the full path, then verifies the result is within the model base directory.

This catches `..` traversal and symlinks in any path component (not just the final one).

### Layer 2: Symlink Detection

- **C++**: `std::filesystem::is_symlink(data_path)` rejects the final-component symlink.
- **Python**: `os.path.islink(path)` rejects the final-component symlink.

This is a belt-and-suspenders check alongside containment. It provides a clear, specific error message when the final path component is a symlink.

### Layer 3: O_NOFOLLOW on File Open (Python only)

- **Python**: `os.O_NOFOLLOW` added to `os.open()` flags where available (`hasattr(os, "O_NOFOLLOW")`).

The C++ checker validates paths but does not open files, so `O_NOFOLLOW` is not applicable there. In Python, this is the last-resort defense: even if a symlink is created between the check and the open (TOCTOU race), the kernel rejects the open with `ELOOP` on Linux/macOS.

### Layer 4: Hardlink Count Check

- **C++**: `std::filesystem::hard_link_count(data_path) > 1` rejects files with multiple hardlinks.
- **Python**: `os.stat(path).st_nlink > 1` rejects files with multiple hardlinks.

This prevents an attacker from using a hardlink (which is not a symlink) to point external data at a sensitive file. Note that `O_NOFOLLOW` does **not** protect against hardlinks — only this explicit check does.

## Protected Entry Points

Not all layers apply at every entry point. The C++ checker validates paths but does not open files, so Layer 3 (O_NOFOLLOW) is Python-only.

| Entry Point | File | Layers |
|---|---|---|
| `_resolve_external_data_location` | `onnx/checker.cc` | 1, 2, 4 |
| `load_external_data_for_tensor` | `onnx/external_data_helper.py` | 1, 2, 3, 4 |
| `save_external_data` | `onnx/external_data_helper.py` | 1, 2, 3, 4 |
| `ModelContainer._load_large_initializers` | `onnx/model_container.py` | 1, 2, 3, 4 |

The C++ checker runs first for all Python load paths (via `c_checker._resolve_external_data_location`). The Python checks serve as defense-in-depth.

## Known Limitations

### TOCTOU (Time-of-Check-to-Time-of-Use)

There is an inherent race window between the security checks (Layers 1-2, 4) and the file open (Layer 3). An attacker with write access to the model directory could:

1. Place a legitimate file to pass checks.
2. Replace it with a symlink or hardlink between the check and the open.

**Mitigation**: `O_NOFOLLOW` (Layer 3) catches late symlink replacement on Linux/macOS at the kernel level. However, `O_NOFOLLOW` does **not** protect against hardlink replacement — this TOCTOU gap cannot be fully closed at the application level.

### Windows

- `O_NOFOLLOW` is **not available** on Windows (`hasattr(os, "O_NOFOLLOW")` returns `False`). The TOCTOU window for symlink attacks is fully open on Windows, relying solely on Layers 1-2.
- Symlink and hardlink tests are skipped on Windows in the test suite.

### Case-Insensitive Filesystems

The canonical path containment check uses string comparison. On case-insensitive filesystems (Windows NTFS, macOS HFS+), paths with different casing may incorrectly fail containment. This fails closed (false rejection, not a bypass).

## Testing

Test coverage is in:

- **C++**: `onnx/test/cpp/checker_test.cc` — `SymLink*` tests for symlink detection and containment.
- **Python**: `onnx/test/test_external_data.py`:
  - `TestSaveExternalDataSymlinkProtection` — save-side symlink rejection.
  - `TestLoadExternalDataSymlinkProtection` — load-side symlink rejection, parent-directory symlink, `load_external_data_for_model` rejection.
  - `TestLoadExternalDataHardlinkProtection` — load-side hardlink rejection.
  - `TestSaveExternalDataAbsolutePathValidation` — absolute path rejection.

Symlink and hardlink tests are skipped on Windows (`os.name == "nt"`).
