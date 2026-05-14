<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX CI Pipelines

## Core CI

| Workflow | When it runs | What it does |
|---|---|---|
| [CI](/.github/workflows/main.yml) | Every PR, merge_group, push to main, daily (midnight UTC) | C++ and Python tests across Linux, Windows, macOS; Python 3.10–3.14 (including free-threading variants); doc generation; proto generation; node test generation; daily run reports code coverage to Codecov |
| [Windows\_No\_Exception\_CI](/.github/workflows/win_no_exception_ci.yml) | Push and PR to main and rel-\* | C++ tests compiled without exceptions; selective schema loading |
| [Lint / Enforce style](/.github/workflows/lint.yml) | Every PR | Required — runs lintrunner (ruff, mypy, clang-format, etc.) and verifies auto-generated files are up to date |
| [Require label](/.github/workflows/check_pr_label.yml) | Every PR | Requires at least one `topic:` or `module:` label (skipped for Dependabot PRs) |
| [DCO](/.github/workflows/dco_merge_group.yml) | merge\_group | Placeholder DCO job required to enable the GitHub merge queue |

## Release Builds (1)

| Workflow | When it runs | What it does |
|---|---|---|
| [Create Releases](/.github/workflows/create_release.yml) | Push to main/rel-\*, PRs targeting rel-\* or labeled "run release CIs", weekly (Monday 00:00 UTC), workflow\_dispatch | Orchestrator — calls WindowsRelease, LinuxRelease, MacRelease, PyodideRelease, and sdistRelease as reusable workflows |
| [WindowsRelease](/.github/workflows/release_windows_cibw.yml) | Called by Create Releases | Builds Windows wheels for x64, x86, and arm64; verifies with min and latest numpy/protobuf; verifies with latest ONNX Runtime PyPI package (2)(3) |
| [LinuxRelease](/.github/workflows/release_linux_cibw.yml) | Called by Create Releases | Builds Linux wheels for x86\_64 (manylinux\_2\_28) and aarch64; verifies with min and latest numpy/protobuf; verifies with latest ONNX Runtime PyPI package |
| [MacRelease](/.github/workflows/release_macos_cibw.yml) | Called by Create Releases | Builds macOS wheels (macos-14, MACOSX\_DEPLOYMENT\_TARGET=12.0); verifies with min and latest numpy/protobuf; verifies with latest ONNX Runtime PyPI package; tests source distribution build |
| [PyodideRelease](/.github/workflows/release_pyodide_cibw.yml) | Called by Create Releases and on every push | Builds a Pyodide (WebAssembly) wheel on Ubuntu using `cibuildwheel` with a pre-downloaded host `protoc` and protobuf source; runs a basic import test |
| [sdistRelease](/.github/workflows/release_sdist.yml) | Called by Create Releases | Builds and tests source distribution |

## Security and Supply Chain

| Workflow | When it runs | What it does |
|---|---|---|
| [CodeQL](/.github/workflows/codeql.yml) | Every PR, push to main/rel-\*, weekly (Friday) | Static analysis of C++ and Python for security vulnerabilities |
| [Scorecard](/.github/workflows/scorecard.yml) | Push to main, weekly (Saturday) | OpenSSF supply-chain security scorecard; publishes results to code-scanning dashboard |
| [Dependency Review](/.github/workflows/dependency-review.yml) | Every PR | Flags vulnerable or license-incompatible dependencies introduced by a PR |

## Documentation and Maintenance

| Workflow | When it runs | What it does |
|---|---|---|
| [Pages](/.github/workflows/pages.yml) | PRs to main, push to main | Builds and publishes ONNX documentation to GitHub Pages |
| [Pixi CI](/.github/workflows/pixi_build.yml) | Weekly (Sunday 23:59 UTC) and on PRs | Builds, lints, and tests with the [pixi](https://pixi.sh/) environment manager on Linux, macOS, and Windows; opens an issue on failure when scheduled |
| [Check URLs](/.github/workflows/check_urls.yml) | Push to main/rel-\*, monthly | Checks for broken URLs in the codebase |
| [Stale](/.github/workflows/stale.yml) | Daily | Warns and eventually closes stale issues and PRs |
| [Dependabot](/.github/dependabot.yml) | Monthly | Creates PRs for updated dependency versions |

---

* **(1)** Release CIs run when:
  * A PR is merged into main or a rel-\* branch
  * Weekly (Monday 00:00 UTC) — publishes a Python wheel to the [onnx-weekly](https://pypi.org/project/onnx-weekly/) package on PyPI
  * Any PR targeting a rel-\* branch
  * Any PR labeled "run release CIs" (maintainers only)
  * Manually via workflow\_dispatch

* **(2)** Minimum supported dependency versions are listed in `[project.dependencies]` in [pyproject.toml](/pyproject.toml).
