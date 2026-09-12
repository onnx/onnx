<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Release Administration

This document covers privileged maintenance tasks associated with ONNX releases.
These tasks are owned by administrators in the Architecture & Infra SIG and are
not part of the release manager's release checklist.

## PyPI Storage Cleanup

Package deletion is irreversible. Before deleting a distribution, verify the
project, version, package type, and target package index, and coordinate the
cleanup with the Architecture & Infra SIG.

### Weekly Packages

After a stable ONNX release, administrators may remove
[`onnx-weekly`](https://pypi.org/project/onnx-weekly/#history) distributions for
the version that was just released to conserve project storage.

1. Open the [onnx-weekly release management page](https://pypi.org/manage/project/onnx-weekly/releases/).
2. Select the obsolete release and verify its version and files.
3. Use **Options > Delete** to remove it.

The `onnx-weekly` and `onnx` projects have separate access control. Request
access from an existing project owner when necessary.

### TestPyPI Release Candidates

After the corresponding stable release and partner validation are complete,
administrators may remove obsolete ONNX release-candidate distributions from
TestPyPI. Retain candidates that are still needed to investigate the current
release.

1. Open the [ONNX TestPyPI release management page](https://test.pypi.org/manage/project/onnx/releases/).
2. Select the obsolete release candidate and verify its version and files.
3. Use **Options > Delete** to remove it.
