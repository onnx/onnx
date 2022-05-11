<!--- SPDX-License-Identifier: Apache-2.0 -->

## ONNX CI Pipelines

* CI pipelines matrix:

  |   | When it runs | Config | Test |
  -- | -- | -- | -- | 
  [Linux-CI](/.azure-pipelines/Linux-CI.yml) | Every PR | <ul><li>Ubuntu-18.04</li><li>DEBUG=1 or 0</li><li>ONNX_USE_LITE_PROTO=ON or OFF</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_BUILD_TESTS=1</li><li>ONNX_WERROR=ON</li><li>ONNX_ML=1 or 0</li></ul>| <ul><li>ONNX C++ tests</li><li>Style check (flake8 and mypy)</li><li>Test doc generation</li><li>Test proto generation</li><li>Verify backend node data</li><li>Verify node test generation</li></ul> | 
  [Windows-CI](/.azure-pipelines/Windows-CI.yml) | Every PR  | <ul><li>vs2017-win2016</li><li>ONNX_USE_LITE_PROTO=ON</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=ON</li><li>ONNX_BUILD_TESTS=1</li><li>ONNX_WERROR=ON</li><li>ONNX_ML=1 or 0</li></ul>| <ul><li>Test building ONNX in conda environment</li><li>Test doc generation</li><li>Test proto generation</li></ul> |
  [Mac-CI](/.azure-pipelines/MacOS-CI.yml) | Every PR  | <ul><li>macOS-10.14</li><li>DEBUG=1</li><li>ONNX_USE_LITE_PROTO=ON or OFF</li><li>ONNX_ML=1 or 0</li><li>ONNX_BUILD_TESTS=1</li><li>ONNX_WERROR=ON</li></ul>| <ul><li>ONNX C++ tests</li><li>Test doc generation</li><li>Test proto generation</li></ul>|
  [Windows_No_Exception CI](/.github/workflows/win_no_exception_ci.yml) | Every PR  | <ul><li>vs2019-winlatest</li><li>ONNX_DISABLE_EXCEPTIONS=ON</li><li>ONNX_USE_LITE_PROTO=ON</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li><li>ONNX_USE_MSVC_STATIC_RUNTIME=ON</li><li>ONNX_DISABLE_STATIC_REGISTRATION=ON or OFF</li></ul>| <ul><li>Only ONNX C++ tests</li><li>Test selective schema loading</li></ul> |
  [WindowsRelease](/.github/workflows/release_win.yml) | <ul><li>Main branch</li><li>Release branch</li><li>Weekly(1)</li></ul> | <ul><li>Latest Windows</li><li>x86 and x64</li><li>ONNX_USE_LITE_PROTO=ON</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li><li>ONNX_USE_MSVC_STATIC_RUNTIME=OFF</li></ul>| <ul><li> Release Windows wheel</li><li>Release onnx-weekly package</li><li>Verify backend node data</li><li>Verify node test generation</li><li>Verify with different dependency versions - latest numpy version, latest and min supported protobuf version(2)</li><li>Verify ONNX with the latest [ort-nightly](https://test.pypi.org/project/ort-nightly/)(3).</li></ul> |
  [LinuxRelease_aarch64](/.github/workflows/release_linux_aarch64.yml) | <ul><li>Main branch</li><li>Release branch</li><li>Weekly</li></ul>  | <ul><li>Latest manylinux2014_aarch64</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li></ul>| <ul><li> Release Linux aarch64 wheel</li><li>Release onnx-weekly package</li><li>Verify backend node data</li><li>Verify node test generation</li><li>Verify with different dependency versions - latest numpy version, latest and min supported protobuf version</li><li>Verify ONNX with the latest ort-nightly.</li></ul> |
  [LinuxRelease_x86_64](/.github/workflows/release_linux_x86_64.yml) | <ul><li>Main branch</li><li>Release branch</li><li>Weekly</li></ul> | <ul><li>Latest LinuxRelease_x86_64</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li></ul>| <ul><li> Release Linux x86_64 wheel</li><li>Release onnx-weekly package</li><li>Test TEST_HUB=1(4)</li><li>Verify backend node data</li><li>Verify node test generation</li></li><li>Verify with different dependency versions - latest numpy version, latest and min supported protobuf version</li><li>Verify ONNX with the latest ort-nightly.</li></ul> |
  [MacRelease](/.github/workflows/release_win.yml) | <ul><li>Main branch</li><li>Release branch</li><li>Weekly</li></ul> | <ul><li>macos-10.15</li><li> MACOSX_DEPLOYMENT_TARGET=10.12(5) </li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li></ul>| <ul><li>Release Mac wheel</li><li>Release onnx-weekly package</li><li>Verify backend node data</li><li>Verify node test generation</li><li>Verify with different dependency versions - latest numpy version, latest and min supported protobuf version</li><li>Verify ONNX with the latest ort-nightly.</li><li>Test source distribution generation</li><li>Test build with source distribution</li><li>Release onnx-weekly source distribution</li></ul>  
  [Weekly CI with latest onnx.checker](/.github/workflows/weekly_mac_ci.yml) | weekly(6) |<ul><li>macos-latest</li><li>MACOSX_DEPLOYMENT_TARGET=10.15</li><li>ONNX_USE_PROTOBUF_SHARED_LIBS=OFF</li><li>ONNX_ML=1</li></ul>| <ul><li>Test latest ONNX checker</li><li>Test latest ONNX shape inference</li><li>With all models from [onnx/models](https://github.com/onnx/models)(7)</li></ul> |
  
  * (1) When the release CIs will run:
    * After a PR has been merged into main/rel-* branch
    * Run weekly (Sunday midnight) and release Python wheel to [onnx-weekly](https://test.pypi.org/project/onnx-weekly/) package on TestPyPI.
    * Any PR targeting rel-* branch
    * To manually run them, add a PR label "run release CIs" (only maintainers have permission).
  * (2) Minimum supported versions are listed [here](/requirements.txt).  
  * (3) [Test](/onnx/test/test_with_ort.py) ONNX Python wheel with `onnxruntime.InferenceSession` from latest ONNXRuntime. Please note that ort-nightly does not support Windows-x86 thus its verification is skipped.  
  * (4) TEST_HUB=1 will test [onnx.hub](/onnx/test/hub_test.py) by using this API to download an ONNX model from onnx/models. This test is restricted to only 1 pipeline for saving quota usage.
  * (5) Although the build environment is macos-10.15, use MACOSX_DEPLOYMENT_TARGET=10.12 and -p [macosx_10_12_x86_64](https://github.com/onnx/onnx/blob/2e048660ffa8243596aaf3338e60c7c0575458f2/.github/workflows/release_mac.yml#L74) to force the wheel to support 10.12+.  
  
  * (6):
    * The ONNX Model Zoo test will run weekly (Sunday midnight)
    * To manually trigger it, add a PR label "test ONNX Model Zoo" (only maintainers have permission). Please note that it will need a lot of download bandwidth from [onnx/models](https://github.com/onnx/models) so use it with caution.
  * (7) Some old deprecated models (opset-1) are [skipped](/workflow_scripts/config.py).
