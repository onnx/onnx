<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX CI Pipelines

## Core CI

| Workflow | When it runs | What it does |
|---|---|---|
| [CI](/.github/workflows/main.yml) | Every PR, merge_group, push to main, daily (midnight UTC), workflow_dispatch | C++ and Python tests across Linux, Windows, macOS using the configured Python matrix (including free-threading variants); doc generation; proto generation; node test generation; daily run reports code coverage to Codecov |
| [Windows\_No\_Exception\_CI](/.github/workflows/win_no_exception_ci.yml) | Push and PR to main and rel-\* | C++ tests compiled without exceptions; selective schema loading |
| [Lint / Enforce style](/.github/workflows/lint.yml) | Every PR | Required — runs lintrunner (ruff, mypy, clang-format, etc.) and verifies auto-generated files are up to date |
| [Require label](/.github/workflows/check_pr_label.yml) | Every PR | Requires at least one `topic:` or `module:` label (skipped for Dependabot PRs) |
| [DCO](/.github/workflows/dco_merge_group.yml) | merge\_group | Placeholder DCO job required to enable the GitHub merge queue |

## Release Builds (1)

| Workflow | When it runs | What it does |
|---|---|---|
| [Create Releases](/.github/workflows/create_release.yml) | Push to main/rel-\*, labeled or synchronize PR events targeting main/rel-\* (jobs gated by labels), weekly (Monday 00:00 UTC), workflow\_dispatch | Orchestrates wheel and sdist builds; publication is controlled separately by dispatch inputs or the weekly schedule |
| [WindowsRelease](/.github/workflows/release_windows_cibw.yml) | Called by Create Releases | Builds Windows wheels for x64, x86, and arm64; runs wheel tests and content/ABI validation (2)(3) |
| [LinuxRelease](/.github/workflows/release_linux_cibw.yml) | Called by Create Releases | Builds Linux wheels for x86\_64 (manylinux\_2\_28) and aarch64; runs wheel tests and content/ABI validation (2)(3) |
| [MacRelease](/.github/workflows/release_macos_cibw.yml) | Called by Create Releases | Builds universal2 macOS wheels on macos-14 with MACOSX\_DEPLOYMENT\_TARGET=13.3; runs wheel tests and content/ABI validation (2)(3) |
| [Pyodide Build](/.github/workflows/release_pyodide_cibw.yml) | Called by Create Releases or workflow\_dispatch | Builds a Python 3.14 Pyodide (WebAssembly) wheel on Ubuntu using `cibuildwheel`, a host `protoc`, and protobuf source; runs a basic import test (3) |
| [sdistRelease](/.github/workflows/release_sdist.yml) | Called by Create Releases | Builds and uploads the source-distribution artifact |
| [Test source distribution build](/.github/workflows/preview_source_dist_test.yml) | Tuesday schedule, workflow\_dispatch, or selected post-publication calls from Create Releases | Builds and tests an sdist from the checkout on Linux/Windows; does not download and verify the published sdist |

## Security and Supply Chain

| Workflow | When it runs | What it does |
|---|---|---|
| [CodeQL](/.github/workflows/codeql.yml) | PRs targeting main and pushes to main (excluding docs-only changes), weekly (Friday), workflow_dispatch | Static analysis of Actions, C++, and Python |
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
  * Labeled/synchronize events for PRs targeting main or rel-\*: apply `run release CIs` for the full release build. Targeting a release branch alone does not bypass the orchestrator's label gate.
  * The `ci-slsa-provenance` label enables the provenance path, subject to each reusable workflow's own job conditions.
  * Manually via workflow\_dispatch

  Push and PR builds do not publish stable packages automatically. See the
  [release checklist](OnnxReleases.md#build-and-publish-packages) for the exact
  branch, build-mode, publish-input, and deployment-environment combinations.

* **(2)** Python requirements and minimum dependency versions are declared in
  [pyproject.toml](/pyproject.toml); the exact wheel matrix is in each workflow.
  The configured wheel tests do not pin every dependency to its minimum version,
  so passing them is not proof of testing all dependency lower bounds.

* **(3)** The [PEP 770](https://peps.python.org/pep-0770/) SBOM (`dist-info/sboms/sbom.cdx.json`) is embedded in any wheel build where `SKBUILD_METADATA_DIR` is set (including local `pip wheel` builds). Only the cibuildwheel release pipeline patches the SBOM to reflect the actual protobuf tarball version, URL, and SHA-256 used for that specific build; other wheel builds embed the template values from `sbom.cdx.json`.
