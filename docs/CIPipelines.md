<!--- SPDX-License-Identifier: Apache-2.0 -->

## ONNX CI Pipelines

* CI pipelines matrix:

  |   | Kind | When | Purposes |
  -- | -- | -- | -- |
  [Linux-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/Linux-CI.yml) | Azure Pipelines  | Any PR(1)  |   <ul><li>Ubuntu-18.04</li><li>Test DEBUG=1</li><li>Test Protobuf-Lite</li><li>ONNX C++ tests</li><li>Test flake8</li><li>Test doc generation</li><li>Test proto generation</li><li>mypy typecheck</li><li>Verify uploaded node models</li><li>Verify node test generation</li></ul>|  | 
  [Windows-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/Windows-CI.yml) | Azure Pipelines  | Any PR  |  <ul><li>vs2017-win2016</li><li>Test Protobuf-Lite</li><li>Test Conda</li><li>Test doc generation</li><li>Test proto generation</li><li>mypy typecheck</li></ul>| |
  [Mac-CI](https://github.com/onnx/onnx/blob/master/.azure-pipelines/MacOS-CI.yml) | Azure Pipelines | Any PR  |  <ul><li>macOS-10.14</li><li>Test DEBUG=1</li><li>Test Protobuf-Lite</li><li>ONNX C++ tests</li><li>Test flake8</li><li>Test doc generation</li><li>Test proto generation</li></ul>  | 
  [Windows_No_Exception_CI](https://github.com/onnx/onnx/blob/master/.github/workflows/win_no_exception_ci.yml)  | GitHub Action  | PR  | <ul><li>vs2019-winlatest</li><li>Test no-exception build (ONNX_DISABLE_EXCEPTIONS=ON)</li><li>Test Protobuf-Lite</li><li>ONNX C++ tests</li><li>Test selective schema loading</li></ul>  |
  [WindowsRelease](https://github.com/onnx/onnx/blob/master/.github/workflows/release_win.yml)  | GitHub Action  | branch/weekly | <ul><li>latest Windows</li><li> Test x86 and x64</li><li> Release Windows wheel</li><li>Release onnx-weekly package</li><li>Verify uploaded node models</li><li>Verify node test generation</li><li>Verify ONNX with the latest NumPy</li><li>Verify ONNX with the latest Protobuf</li><li>Verify ONNX with minimum supported Protobuf(5)</li><li>Verify ONNX with the latest [ort-nightly](https://test.pypi.org/project/ort-nightly/)(6).</li></ul>|
  [LinuxRelease_aarch64](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_aarch64.yml)  | GitHub Action | branch/weekly(2)  | <ul><li>latest manylinux2014_aarch64</li><li> Release Linux aarch64 wheel</li><li>Release onnx-weekly package</li><li>Verify uploaded node models</li><li>Verify node test generation</li><li>Verify ONNX with the latest NumPy</li><li>Verify ONNX with the latest Protobuf</li><li>Verify ONNX with minimum supported Protobuf</li><li>Verify ONNX with the latest ort-nightly.</li> |
  [LinuxRelease_i686](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_i686.yml)  | GitHub Action  |  branch/weekly  | <ul><li>latest manylinux2010_x86_64</li><li> Release Linux i686 wheel</li><li>Release onnx-weekly package</li><li>Verify uploaded node models</li><li>Verify node test generation</li><li>Verify ONNX with the latest NumPy</li><li>Verify ONNX with the latest Protobuf</li><li>Verify ONNX with minimum supported Protobuf</li></li> |
  [LinuxRelease_x86_64](https://github.com/onnx/onnx/blob/master/.github/workflows/release_linux_x86_64.yml)  | GitHub Action | branch/weekly | <ul><li>latest manylinux2014_aarch64</li><li> Release Linux x86_64 wheel</li><li>Release onnx-weekly package</li><li>Test TEST_HUB=1(7)</li><li>Verify uploaded node models</li><li>Verify node test generation</li><li>Verify ONNX with the latest NumPy</li><li>Verify ONNX with the latest Protobuf</li><li>Verify ONNX with minimum supported Protobuf</li><li>Verify ONNX with the latest ort-nightly.</li>|
  [MacRelease](https://github.com/onnx/onnx/blob/master/.github/workflows/release_win.yml)  | GitHub Action  | branch/weekly | <ul><li>macos-10.15</li><li> MACOSX_DEPLOYMENT_TARGET=10.12(8) </li><li> Release Mac wheel</li><li>Release onnx-weekly package</li><li>Verify uploaded node models</li><li>Verify node test generation</li><li>Verify ONNX with the latest NumPy</li><li>Verify ONNX with the latest Protobuf</li><li>Verify ONNX with minimum supported Protobuf</li><li>Verify ONNX with the latest ort-nightly.</li><li>Test source distribution generation</li><li>Test build with source distribution</li><li>Release onnx-weekly source distribution</li></ul>|  
  [Weekly CI with latest onnx.checker](https://github.com/onnx/onnx/blob/master/.github/workflows/weekly_mac_ci.yml)   | GitHub Action  |  weekly(3) |  <ul><li>Test latest ONNX checker</li><li>Test latest ONNX shape inference</li><li>With all models from [onnx/models](https://github.com/onnx/models)(4)</li></ul> |  
  
  * (1): It will be run by every PR.
  * (2):
    * After a PR merges into main/rel-* branch, the CI will be run.
    * These release CIs will be run weekly (Sunday midnight) and release Python wheel to [onnx-weekly](https://test.pypi.org/project/onnx-weekly/) package in TestPyPI.
    * The PR to merge into rel-* branch will be run because they are supposed to be released soon.
    * To manually run it, add PR label "run release CIs" (only maintainers have permission).
  * (3):
    * The ONNX Model Zoo test will be run weekly (Sunday midnight)
    * To manually run it, add PR label "test ONNX Model Zoo" (only maintainers have permission). Please note that it will need a lot of download bandwidth from [onnx/models](https://github.com/onnx/models) so use it with cautious.
  * (4) Some old deprecated models (opset-1) are [skipped](https://github.com/onnx/onnx/blob/master/workflow_scripts/config.py).
  * (5) Minimum supported versions are listed [here](https://github.com/onnx/onnx/blob/master/requirements.txt).
  * (6) [Test](https://github.com/onnx/onnx/blob/master/onnx/test/test_with_ort.py) ONNX Python wheel with `onnxruntime.InferenceSession` from latest ONNXRuntime. Please note that ort-nightly does support Linux-i686 and Windows-x86 thus their verification are skipped.
  * (7) TEST_HUB=1 will test [onnx.hub](https://github.com/onnx/onnx/blob/master/onnx/test/hub_test.py) by using this API to download a ONNX model from onnx/models.
  * (8) Although the build envioronment is macos-10.15, use MACOSX_DEPLOYMENT_TARGET=10.12 and -p [macosx_10_12_x86_64](https://github.com/onnx/onnx/blob/2e048660ffa8243596aaf3338e60c7c0575458f2/.github/workflows/release_mac.yml#L74) to force the wheel to support 10.12+.
