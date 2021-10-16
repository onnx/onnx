<!--- SPDX-License-Identifier: Apache-2.0 -->

## ONNX CI Pipelines

* CI pipelines matrix:

  |   | Kind | When | Purpose |
  -- | -- | -- | -- |
  [Linux-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/Linux-CI.yml) | Azure Pipeline  | PR(1)  |   <ul><li>Ubuntu-18.04</li><li>Test DEBUG=1</li><li>Test ONNX_ML=1</li><li>Test Protobuf-Lite</li><li>ONNX C++ tests</li><li>Test flake8</li><li>Test doc generation</li><li>Test proto generation</li><li>mypy typecheck</li><li>Verify node test generation</li></ul>|  | 
  [Windows-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/Windows-CI.yml) | Azure Pipeline  | PR  |  <ul><li>Test ONNX_ML=1</li><li>Test Protobuf-Lite</li><li>Test Conda</li><li>Test doc generation</li><li>Test proto generation</li><li>mypy typecheck</li></ul>| |
  [Mac-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/MacOS-CI.yml) | Azure Pipeline | PR  |  <ul><li>macOS-10.14</li><li>Test DEBUG=1</li><li>Test ONNX_ML=1</li><li>Test Protobuf-Lite</li><li>ONNX C++ tests</li><li>Test flake8</li><li>Test doc generation</li><li>Test proto generation</li></ul>  | 
  [Windows_No_Exception_CI](https://github.com/onnx/onnx/blob/master/.github/workflows/win_no_exception_ci.yml)  | GitHub Action  | PR  |   |   | 
  [LinuxRelease_aarch64](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_aarch64.yml)  | GitHub Action | branch/weekly(2)  |   |   |
  [LinuxRelease_i686](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_i686.yml)  | GitHub Action  |  branch/weekly  |   |   | 
  [LinuxRelease_x86_64](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_x86_64.yml)  |  GitHub Action |   branch/weekly |   |   | 
  [WindowsRelease](https://github.com/onnx/onnx/blob/master/.github/workflows/release_win.yml)  | GitHub Action  |   branch/weekly |   |   | 
  [Weekly CI with latest onnx.checker](https://github.com/onnx/onnx/blob/master/.github/workflows/weekly_mac_ci.yml)   | GitHub Action  |  weekly(3) |   |  
  
  * (1): It will be run by every PR.
  * (2):
    * After a PR merges into main/rel-* branch, the CI will be run.
    * These release CIs will be run weekly (Sunday midnight) and release Python wheel to [onnx-weekly](https://test.pypi.org/project/onnx-weekly/) package in TestPyPI.
    * The PR to merge into rel-* branch will be run because they are supposed to be released soon.
    * To manually run it, add PR label "run release CIs" (only maintainers have permission).
  * (3):
    * The ONNX Model Zooo test will be run weekly (Sunday midnight)
    * To manually run it, add PR label "test ONNX Model Zoo" (only maintainers have permission). Please note that it will need a lot of download bandwidth from [onnx/models](https://github.com/onnx/models) so use it with cautious.

