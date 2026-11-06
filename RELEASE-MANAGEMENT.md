<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX release management

This document describes ONNX release policy, cadence, and compatibility. The
operational procedure for release managers is maintained separately in the
[ONNX release checklist](docs/OnnxReleases.md).

Release Cadence
---------------

Branch cuts for a new release are planned every 3 months. However, the times can be changed as required.

| Minor Version | Release branch cut | Release date | Patch release dates |
| --- | --- | --- | --- |
| 1.17.0 | — | 30 September 2024 | Not planned |
| 1.18.0 | March 2025 | 8 May 2025 | Not planned |
| 1.19.0 | 31 July 2025 | 26 August 2025 | 7 October 2025 |
| 1.20.0 | 31 October 2025 | 1 December 2025 | 9 January 2026 |
| 1.21.0 | 25 February 2026 | 27 March 2026 | — |
| 1.22.0 | 1 June 2026 | 15 June 2026 | — |
| 1.23.0 | 1 September 2026 | 18 September 2026 | 29 September 2026 (1.23.1), 6 October 2026 (1.23.2) |
| 1.24.0 (planned) | 1 December 2026 | 14 December 2026 | TBD |
| 1.25.0 (planned) | 1 February 2027 | 28 February 2027 | TBD |

Dates for unpublished releases are targets and may change. Published release
dates are recorded on the [GitHub Releases page](https://github.com/onnx/onnx/releases).

## Long-Term Support (LTS)

ONNX does not currently maintain LTS branches. Each minor release is superseded by the next; only the latest release receives ongoing attention. If your environment cannot upgrade and you need a backported security or bug fix, please
[open an issue](https://github.com/onnx/onnx/issues/new) describing your situation — we will assess feasibility on a case-by-case basis. Community contributions for backport PRs are welcome and will be reviewed.

Release Compatibility Matrix
----------------------------

*Support for a Python version that has reached end of life will be discontinued
in the following ONNX release.*

*ONNX does not follow the
[Scientific Python ecosystem support policy](https://scientific-python.org/specs/spec-0000/)
or the [Protobuf support policy](https://protobuf.dev/support/version-support/).*

Changes are discussed in the community. Please do not hesitate to contact us if you have any requests.
Planned changes for future releases as listed in the table below are subject to change.

|ONNX version	| Python wheels | C++ | Min CMake Version | Min Protobuf | manylinux |
| --- | --- | --- | --- | --- | --- |
| 1.10 | 3.6-3.9 | 11 | 3.1 | --- | manylinux2010 |
| 1.11 | 3.6-3.9 | 11 | 3.1 | 3.12.2 | manylinux2010 |
| 1.12 | 3.7-3.10 | 11 | 3.1 | 3.12.2 | manylinux2014 |
| 1.13 | 3.7-3.11 | 11 | 3.1 | 3.20.2 | manylinux2014 |
| 1.14 | 3.7-3.11 | 11 | 3.1 | 3.20.2 | manylinux2014 |
| 1.15 | 3.8-3.11 | 14 | 3.1 | 3.20.2 | manylinux2014 |
| 1.16 | 3.8-3.12 | 17 | 3.1 | 3.20.2 | manylinux2014 |
| 1.17 | 3.8-3.12 | 17 | 3.14 | 3.20.2 | manylinux2014 |
| 1.18 | 3.9-3.13, 3.13t (win, mac) | 17 | 3.18 | v25.1 | manylinux2014 |
| 1.19 | 3.9-3.13, 3.13t (win, mac, linux) | 17 | 3.24 | v25.1 | manylinux2014 |
| 1.19.1 | 3.9-3.13, 3.13t (win, mac, linux) | 17 | 3.24 | v25.1 | manylinux2014 |
| 1.20 | 3.10-3.13, 3.13t (win, mac, linux), 3.14 (mac) | 17 | 3.26 | v25.1 | manylinux_2_28 |
| 1.21 | 3.10-3.13, 3.13t, 3.14, 3.14t | 17 | 3.26 | v25.1 | manylinux_2_28 |
| 1.22 | 3.10, 3.11, 3.12+ (abi3), 3.14t, 3.14 Pyodide | 17 | 3.24 | v25.1 | manylinux_2_28 |
| 1.23 | 3.10, 3.11, 3.12+ (abi3), 3.14t, 3.14 Pyodide | 17 | 3.26 | v31.1 | manylinux_2_28 |
| *1.24 (planned)* | 3.11, 3.12+ (abi3), 3.14t, 3.15t, 3.14 Pyodide | 17 | 3.26 | v31.1 | manylinux_2_28 |

`abi3` wheels built for CPython 3.12 are intended for CPython 3.12 and later.
Free-threaded builds are shown with the `t` suffix and require separate wheels.
Planned compatibility reflects the current build configuration and may change
before release.


Releases
--------

Releases are versioned according to [ONNX Versioning](docs/Versioning.md). This
describes IR and operator versioning policies and proposes how models themselves
should be versioned.

On a regular basis, new versions of ONNX are published, representing the aggregate of changes in the IR and operator sets. Such releases use semantic versioning to describe the progression of the standard.

The GitHub repo for ONNX provides release branches where the project is stabilized as per the process described here. Release notes are used to communicate the stability and status of a release. The main branch will be used to continue work for subsequent releases.

Major, minor and patch releases will have branch names and version numbers reflecting the nature of the change as per semantic versioning definitions.

Workflow
--------

The authoritative operational procedure is the
[ONNX release checklist](docs/OnnxReleases.md). It covers preparation, creating
the `rel-X.Y.Z` branch, publishing and validating release candidates, creating
the final tag, publishing packages, and post-release tasks. Release candidate
versions use the PEP 440 form `X.Y.ZrcN` (for example, `1.24.0rc1`).

The [Create Releases workflow](.github/workflows/create_release.yml) builds the
platform wheels and source distribution and publishes them through protected
GitHub environments and PyPI Trusted Publishing. The workflow and release
checklist are the sources of truth for the exact build and publication steps.

Testing
-------

The release process really consists of communicating, testing to establish a
known state of the project, and distributing files. This section deals with
the second task.

At the very least, the tests that are part of the `tests/` directory should be run
under a variety of configurations. Issues fixed should ensure coverage in this
suite to avoid regressions.

Send a Pull Request for updates to this section to include a configuration you
can help test if you care about one that's missing.

The community is encouraged to perform additional testing during the test
periods. Bugs and issues should be filed in the ONNX GitHub repo.

# ONNX Weekly Builds on PyPI

In addition to stable releases, we publish **weekly development builds** to a separate PyPI package: [`onnx-weekly`](https://pypi.org/project/onnx-weekly/).

## Why a Separate Package?

- **Avoid accidental installs:** Pre-release versions can be installed unintentionally; `onnx-weekly` ensures stable users are unaffected.
- **Enable isolated testing:** Try upcoming features in a dedicated virtual
  environment without changing a stable environment.
- **Simplify automation:** Weekly builds are pushed automatically from `main` without polluting the main release history.

## Installation

```bash
pip install onnx-weekly
```

`onnx` and `onnx-weekly` are separate PyPI projects, but both install the same
Python import package (`onnx`). Do not install them together in one environment;
use separate virtual environments or uninstall one before installing the other.
