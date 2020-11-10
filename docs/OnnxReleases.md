The ONNX project, going forward, will plan to release roughly on a two month cadence. We follow the [Semver](https://semver.org/) versioning approach and will make decisions as a community on a release by release basis on whether to do a major or minor release.

Next expected release: May 15th 

## Preparation

* Install Twine, a utility tool to interact with PyPI. Do  - ``pip install twine``
* Get hold of the username and password for the ‘onnx’ PyPI account. Release Manager should obtain this information from last Release Manager or certain private repo under onnx.
* Pick a release tag (v.1.X.X) for the new release through mutual consent – Slack channel for Releases (https://lfaifoundation.slack.com/archives/C018VGGJUGK)
* Prepare a change log for the release – 
    * ``git log --pretty=format:"%h - %s" <tag of the previous release>...<new tag>``
    * And draft a new release statement - https://github.com/onnx/onnx/releases listing out the new features and bug fixes, and potential changes being introduced in the release.
* Create a release branch (rel-1.X.X) from master. Checkout the release tag in a clean branch on your local repo. Make sure all tests pass on that branch.
* In the release branch, make sure that the Release version number information is up-to-date in the following places:
[VERSION_NUMBER file](https://github.com/onnx/onnx/blob/master/VERSION_NUMBER) and
[version.h](../onnx/common/version.h)
* Make sure that the IR version number and opset version numbers are up-to-date in
[ONNX proto files](../onnx/onnx.in.proto),
[Versioning.md](Versioning.md), 
[schema.h](../onnx/defs/schema.h), 
[helper.py](../onnx/helper.py) and [helper_test.py](../onnx/helper_test.py).
* Make sure all operators version are even number, if opset version is odd number then increment it and update all corresponding OPs.
* Start early to create the release summary (draft).

## Test with TestPyPI
**Wheel Files**
* Use `VERSION_NUMBER` like `1.x.0rc1` as release candidate for verification before finally using targeted `VERSION_NUMBER`.
* Windows
  * Use GitHub Action (`.github/workflows/release_win.yml`) under onnx repo to produce wheel files for Windows.
  * After success, upload the produced wheel files manually to TestPyPI: `twine upload --verbose *.whl --repository-url https://test.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD`.
* Linux and Mac
  * Use Travis CI from [onnx/wheel-builder](https://github.com/onnx/wheel-builder) repo to produce and automatiallcy upload wheel files for Linux and Mac.
  * Update `BUILD_COMMIT` as the commit ID of latest release branch in `.travis.yml` and update `ONNX_NAMESPACE` as `ONNX_REL_1_X` in `config.sh `.
  * Update `$PYPI_USERNAME` and `$PYPI_PASSWORD` as `- secure: ***` in `.travis.yml`. Create the encrypted variables for these variables by `travis encrypt` in your local machine.
  Reference: https://docs.travis-ci.com/user/environment-variables/#defining-encrypted-variables-in-travisyml
  * Only `PyPI-test` branch will automatiallcy upload created wheel files to TestPyPI.
  * Currently Python 3.5 on Mac cannot upload wheel successfully in Travis CI. In that case, you need to upload the created wheel files to AWS S3 bucket, get the wheel from AWS and upload it manually (same as Windows). To upload to AWS, updade your `ARTIFACTS_KEY`, `ARTIFACTS_SECRET` and `ARTIFACTS_BUCKET` by `travis encrypt`. Reference: https://docs.travis-ci.com/user/uploading-artifacts/

**Source Distribution**
* Make sure all the git submodules are updated
    * ``git submodule update --init``
* Make sure the git checkout is clean –
    * Do ``git clean -nxd`` and make sure that none of the auto-generated header files *like* the following are not present.
        * onnx/onnx-operators.pb.cc
        * onnx/onnx-operator.pb.h
        * onnx/onnx.pb.cc
        * onnx/onnx.pb.h
    * If they are present run ``git clean -ixd`` and remove those files from your local branch
* Do ``python setup.py sdist`` to generate the source distribution.
* Put the following content into ``~/.pypirc`` file:
```
[distutils]
index-servers =
  pypi
  pypitest
 
[pypi]
username=<username>
password=<password>
 
[pypitest]
repository=https://test.pypi.org/legacy/
username=<username>
password=<password>
```
* Do ``twine upload dist/* pypitest`` to upload it to the test instance of PyPI.
* After uploading to PyPItest, you can test the source distribution by doing ``pip install --index-url https://test.pypi.org/legacy/ onnx`` in a new environment. Test this installation with different environments and versions of protobuf binaries.

## PyPI package verification
**Test ONNX itself**
* Test the PyPI package installation with different combinations of various Python versions, Protobuf versions and platforms.
* After installing the PyPI package, run `pytest` in the release branch and test the ONNX basic functions like `load`, `checker.check_model` and `shape_inference.infer_shapes` on certain example ONNX model.

**Test with onnxruntime**
* onnxruntime uses ONNX as submodule so it cannot be tested with ONNX PyPI package. Instead, onnxruntime can be tested with ONNX's release commit. Update the commit ID in onnxruntime and check the status of CIs from onnxruntime.
* To test the interaction with onnxruntime, use ONNX functions like `load`, `checker.check_model`, `shape_inference.infer_shapes`, `save` with onnxruntime functions like `InferenceSession` and `InferenceSession.run` on certain example ONNX model.

**Test with ONNX converters**
* Cooperate with the converter teams. Provide them with the produced ONNX PyPI packages and let converter teams use them with their converters to check whether there is any issue.

## Upload to official PyPI
**NOTE: Once the packages are uploaded to PyPI, you cannot overwrite it on the same PyPI instance. Please make sure everything is good on TestPyPI before uploading to PyPI**

**Wheel Files**
* Windows
  * Same as TestPyPI, use `twine upload --verbose *.whl --repository-url https://upload.pypi.org/legacy/ -u PYPI_USERNAME -p PYPI_PASSWORD` instead.
* Linux and Mac
  * Similar to TestPyPI. Merge `pypi-test` branch to main branch and create a new Release with main branch and tag to trigger Travis CI with uploading PyPI packages automatically. 


**Source Distribution**
* Follow the same process in TestPyPI to produce the source distribution.
* Use ``twine upload --verbose dist/* --repository-url https://upload.pypi.org/legacy/`` instead to upload to the official PyPI.
* Test with ``pip install --index-url https://upload.pypi.org/legacy/ onnx``

## After PyPI Release 

**Release summary**
* Upload the source distribution, `.tar.gz` and `.zip`, in the release summary.
* Submit the release summary with the release tag and branch.

**Announce**
* Announce in slack, for instance, `onnx-general` channel.
* Notify ONNX partners like converter team and runtime team.
* Create a news by updating `js/news.json` to announce ONNX release under [onnx/onnx.github.io](https://github.com/onnx/onnx.github.io) repo.

**Update conda-forge package with the new ONNX version**
* Conda builds of ONNX are done via conda-forge, which runs infrastructure for building packages and uploading them to conda-forge. You need to submit a PR to https://github.com/conda-forge/onnx-feedstock (see https://github.com/conda-forge/onnx-feedstock/pull/1/files for an example PR.) You will need to have uploaded to PyPI already, and update the version number and tarball hash of the PyPI uploaded tarball.

**Set protected branch**
* The release branch needs to be marked as a protected branch to preserve it. The branch corresponding to each release must be preserved. (You may need to ask an admin to do this; `rel-*` branches are protected automatically)


 


