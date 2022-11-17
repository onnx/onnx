# ONNX Release 1.13.0

**Release Manager:** Przemyslaw Wysocki

**Target Release date**: 12/12/2022

## Timetable

* 10/13 - Create [v1.13.0 release wiki](https://github.com/onnx/onnx/wiki/Logistics-for-ONNX-Release-1.13.0) with release schedule.
* 10/13 - Document key v1.13.0 changes for the release in this wiki.
* 11/18 - Code freeze. All PRs must be validated and merged by this date.
* 11/21 - Cut the release branch.
* xx/xx - Create test packages for v1.13.0rc1


### References
* [Drafting ONNX Releases](https://github.com/onnx/onnx/blob/master/docs/OnnxReleases.md)

## Changelog

### New operators introduced in ai.onnx opset 18
#### all new operators are to be validated by TBD
| Op Name | Description |Validation status | ONNX PR 
|---------|:-------------------------:|:------------:|:------------:|
|Col2Im|Rearranges input tensor into blocks||[#3948](https://github.com/onnx/onnx/pull/3948)|
|BitwiseNot|Bitwise operations set||[#4497](https://github.com/onnx/onnx/pull/4497)|
|BitwiseAnd, BitwiseOr and BitwiseXor|Bitwise operations set||[#4496](https://github.com/onnx/onnx/pull/4496)|

### Operator updates in ai.onnx opset 18
#### all updated operators are to be validated by TBD
| Op Name | Description |Validation status | ONNX PR 
|---------|:-------------------------:|:------------:|:------------:|
|Resize|New attributes: `antialias`, `axes` and `keep_aspect_ratio_policy`||[#4126](https://github.com/onnx/onnx/pull/4126)|
|Resize|Allow for both `scales` and `sizes` to be provided when one of them is an empty constant||[#4388](https://github.com/onnx/onnx/pull/4388)|
|Pad|New attribute `axes`||[#4190](https://github.com/onnx/onnx/pull/4190)|
|OptionalHasElement|New input types handling||[#4326](https://github.com/onnx/onnx/pull/4326)|
|OptionalHasElement and OptionalGetElement|Accept tensor and sequence types||[#4421](https://github.com/onnx/onnx/pull/4421)|
|ScatterElement and ScatterND|Add `max` and `min` as supported reduction attributes||[#4411](https://github.com/onnx/onnx/pull/4411)|
|Split|Add support for uneven tensor splitting and new `num_outputs` attribute||[#4481](https://github.com/onnx/onnx/pull/4481)|
|LpPool IN REVIEW|New attributes: `ceil_mode` and `dilations`||[#4534](https://github.com/onnx/onnx/pull/4534)|

### Function updates in ai.onnx opset 18
#### all new and updated operators are to be validated by TBD
| Function Name | Description |Validation status | ONNX PR 
|---------|:-------------------------:|:------------:|:------------:|
|CenterCropPad|New function, center crops or pads an input to given dimensions||[#4190](https://github.com/onnx/onnx/pull/4190)|
|mish|New activation function||[#4350](https://github.com/onnx/onnx/pull/4350)|
|GroupNormalization|New normalization function||[#4621](https://github.com/onnx/onnx/pull/4621)|

### Reference Python runtime
Reference Python runtime dependent on only Python and numpy has been added.
ONNX PR: https://github.com/onnx/onnx/pull/4483

### Python 3.11 support
ONNX now supports Python 3.11.
ONNX PR: https://github.com/onnx/onnx/pull/4490

### Bugfixes and infrastructure improvements
| Description | PR | Status | Notes |      
|--------------------------------------|:------:|:------------:|:------------:|
|Fix concurrent remove of .proto3.* files during build|[#3317](https://github.com/onnx/onnx/pull/3317)|Merged||
|Avoid installing empty directories|[#3590](https://github.com/onnx/onnx/pull/3590)|Merged||
|Fix bf16 Support|[#4193](https://github.com/onnx/onnx/pull/4193)|Merged||
|Make domain check in shape inference consider "ai.onnx" as ""|[#3590](https://github.com/onnx/onnx/pull/3590)|Merged||
|Upgrade python syntax with pyupgrade|[#4212](https://github.com/onnx/onnx/pull/4212)|Merged||
|Use ONNX Runtime PyPI package instead of ort-nightly|[#4219](https://github.com/onnx/onnx/pull/4219)|Merged||
|Update `requirements.txt`|[#4223](https://github.com/onnx/onnx/pull/4223)|Merged||
|Fix mapping TensorProto to NumPy for bfloat16|[#4234](https://github.com/onnx/onnx/pull/4234)|Merged||
|Update `ir.h` to support initializer better|[#4235](https://github.com/onnx/onnx/pull/4235)|Merged||
|Add Truncation Mode to F32->BF16 Conversion Helper (and fix big-endian support)|[#4238](https://github.com/onnx/onnx/pull/4238)|Merged||
|Print utility extension|[#4246](https://github.com/onnx/onnx/pull/4246)|Merged||
|Fix incorrect expected output in windowing and stft tests, and window function function def|[#4249](https://github.com/onnx/onnx/pull/4249)|Merged||
|Fix: Window functions generate incorrect shape for with `symmetric` attribute|[#4256](https://github.com/onnx/onnx/pull/4256)|Merged||
|Handle raw types correctly in `helper.make_tensor`|[#4262](https://github.com/onnx/onnx/pull/4262)|Merged||
|Fix layer normalization's reference outputs|[#4263](https://github.com/onnx/onnx/pull/4263)|Merged||
|Fix sub-graph generation for LN|[#4268](https://github.com/onnx/onnx/pull/4268)|Merged||
|Improve mapping and add more tests for `make_tensor`|[#4270](https://github.com/onnx/onnx/pull/4270)|Merged||
|Remove reference to `numpy.typing`|[#4277](https://github.com/onnx/onnx/pull/4277)|Merged||
|Fix crash: only traverse graph to add symbols if symbol table presents|[#4282](https://github.com/onnx/onnx/pull/4282)|Merged||
|Revert #3979: Checker should validate the node's inputs/outputs have names when Variadic|[#4283](https://github.com/onnx/onnx/pull/4283)|Merged||
|Update `tensor_util.cc` to get uint64_t data easier|[#4285](https://github.com/onnx/onnx/pull/4285)|Merged||
|Refactor `shape_inference_test.py`|[#4302](https://github.com/onnx/onnx/pull/4302)|Merged||
|`extract_model` run time optimization|[#4324](https://github.com/onnx/onnx/pull/4324)|Merged||
|Add `is_constant_initializer` in `ir.h`|[#4336](https://github.com/onnx/onnx/pull/4336)|Merged||
|Specify pairwise conversion behavior for the Cast op|[#4351](https://github.com/onnx/onnx/pull/4351)|Merged||
|Use `RepeatedPtrField::Get` instead of `RepeatedPtrField::operator[]`, to be compatible with protobuf 3.0 API|[#4354](https://github.com/onnx/onnx/pull/4354)|Merged||
|Clarify minimum version of Protobuf required by ONNX|[#4360](https://github.com/onnx/onnx/pull/4360)|Merged||
|Add detailed error message for saving 2GB proto |[#4366](https://github.com/onnx/onnx/pull/4366)|Merged||
|Extend printer to handle non-empty domain|[#4372](https://github.com/onnx/onnx/pull/4372)|Merged||
|Make C++ and Python `check_model` consistent|[#4386](https://github.com/onnx/onnx/pull/4386)|Merged||
|Fix bugs and add `.clang-tidy`|[#4391](https://github.com/onnx/onnx/pull/4391)|Merged||
|Clarify and add helper for making attribute references|[#4393](https://github.com/onnx/onnx/pull/4393)|Merged||
|Use `CMAKE_INSTALL_LIBDIR` to avoid hardcoded path|[#4395](https://github.com/onnx/onnx/pull/4395)|Merged||
|Fix errors reported by flake8 5.0.2|[#4397](https://github.com/onnx/onnx/pull/4397)|Merged||
|Do not allow to read tensor's `external_data` outside the model directory|[#4400](https://github.com/onnx/onnx/pull/4400)|Merged||
|Add `py.typed` to let mypy get stub for type information|[#4412](https://github.com/onnx/onnx/pull/4412)|Merged||
|Modify the annotation: fix the label serial number mistake|[#4426](https://github.com/onnx/onnx/pull/4426)|Merged||
|Format all python code with black and isort - take 2|[#4427](https://github.com/onnx/onnx/pull/4427)|Merged||
|Forbid possible `ir_version=0` in IR|[#4429](https://github.com/onnx/onnx/pull/4429)|Merged||
|Set up VS Code to format python code at save|[#4430](https://github.com/onnx/onnx/pull/4430)|Merged||
|Deprecate ONNXIFI: ONNX Interface for Framework Integration|[#4431](https://github.com/onnx/onnx/pull/4431)|Merged||
|Expose a Python interface for inference functions|[#4409](https://github.com/onnx/onnx/pull/4409)|Merged||
|Ignore the format PR in git blame|[#4433](https://github.com/onnx/onnx/pull/4433)|Merged||
|Remove redundant `static_cast` in ir_pb_converter.cc|[#4437](https://github.com/onnx/onnx/pull/4437)|Merged||
|Make ONNX installable even with spaces in path|[#4473](https://github.com/onnx/onnx/pull/4473)|Merged||
|Fix resize shape inference|[#4448](https://github.com/onnx/onnx/pull/4448)|Merged||
|Add `hasInput` method to `InferenceContext`|[#4451](https://github.com/onnx/onnx/pull/4451)|Merged||
|Make `check_model` with `full_check` always not modify the model in place|[#4456](https://github.com/onnx/onnx/pull/4456)|Merged||
|Use `find_namespace_packages` to get needed subdirectories to silent lots of warnings|[#4457](https://github.com/onnx/onnx/pull/4457)|Merged||
|Primary ops to function milestone 1|[#4458](https://github.com/onnx/onnx/pull/4458)|Merged||
|Primary ops to function milestone 2|[#4512](https://github.com/onnx/onnx/pull/4512)|In progress||
|Use filesystem to load filename to prevent encoding issues on Windows|[#4470](https://github.com/onnx/onnx/pull/4470)|Merged||
|Remove unnecessary import|[#4484](https://github.com/onnx/onnx/pull/4484)|Merged||
|Reduces the number of ignored mypy errors|[#4495](https://github.com/onnx/onnx/pull/4495)|Merged||
|Avoid modifying node when creating test cases|[#4517](https://github.com/onnx/onnx/pull/4517)|Merged||
|Skip version conversion if non-ONNX domain and throw explicit error|[#4521](https://github.com/onnx/onnx/pull/4521)|Merged||
|Relax protobuf version requirement for version below 4 |[#4535](https://github.com/onnx/onnx/pull/4535)|Merged||
|Move mypy config from `setup.cfg` to `pyproject.toml`|[#4542](https://github.com/onnx/onnx/pull/4542)|Merged||
|Fix several issues regarding recent mapping update|[#4551](https://github.com/onnx/onnx/pull/4551)|Merged||
|Fix shebang for `protoc-gen-mypy.sh` script|[#4568](https://github.com/onnx/onnx/pull/4568)|Merged||
|Catch >2GB models in checker|[#2744](https://github.com/onnx/onnx/pull/2744)|Merged||
|Relax protobuf version requirement for version below 4|[#4535](https://github.com/onnx/onnx/pull/4535)|Merged||
|Bump protobuf from 3.16.0 to 3.18.3|[#4544](https://github.com/onnx/onnx/pull/4544)|Merged||
|Move check_is_experimental_op from check_node to check_graph|[#4556](https://github.com/onnx/onnx/pull/4556)|Merged||
|Find python libs only if BUILD_ONNX_PYTHON is set to ON|[#4579](https://github.com/onnx/onnx/pull/4579)|Merged||
|Enable partial data propagation for Resize's sizes|[#4582](https://github.com/onnx/onnx/pull/4582)|Merged||
|Add type inference for CategoryMapper & TreeEnsembleRegressor/Classifier|[#4600](https://github.com/onnx/onnx/pull/4600)|Merged||
|Check the rest of actual input/output type if Variadic|[#4622](https://github.com/onnx/onnx/pull/4622)|Merged||
|Add 32 and 64 bit unsigned integers as tensor types in IR|[#4634](https://github.com/onnx/onnx/pull/4634)|Merged||
|Upgrade pybind11 to 2.9.2 to improve Python 3.11 support|[#4635](https://github.com/onnx/onnx/pull/4635)|Merged||
|Output missing last new line from coverage reporter|[#4636](https://github.com/onnx/onnx/pull/4636)|Merged||
|provide macos universal2 wheel|[#4642](https://github.com/onnx/onnx/pull/4642)|Merged||
|Fix backend test for ReferenceEvaluator with numpy 1.16.6|[#4658](https://github.com/onnx/onnx/pull/4658)|Merged||

### CI improvements
| Description | PR | Status | Notes |      
|--------------------------------------|:------:|:------------:|:------------:|
|Enhance `test_backend_test.py` and simplify backend test CI|[#3393](https://github.com/onnx/onnx/pull/3393)|Merged||
|Weekly test current `version_converter` with all models from ONNX Model Zoo|[#4040](https://github.com/onnx/onnx/pull/4040)|Merged||
|Verify minimum supported numpy in Windows Release CI|[#4279](https://github.com/onnx/onnx/pull/4279)|Merged||
|Create inline lints with github actions|[#4296](https://github.com/onnx/onnx/pull/4296)|Merged||
|Remove stale bot config|[#4315](https://github.com/onnx/onnx/pull/4315)|Merged||
|Enable stale GitHub action|[#4316](https://github.com/onnx/onnx/pull/4316)|Merged||
|Add new Lint Python CI in the CI document|[#4332](https://github.com/onnx/onnx/pull/4332)|Merged||
|Ignore noisy pylint warnings|[#4359](https://github.com/onnx/onnx/pull/4359)|Merged||
|Set up black and isort|[#4361](https://github.com/onnx/onnx/pull/4361)|Merged||
|Test lower protoc version in Linux CI|[#4365](https://github.com/onnx/onnx/pull/4365)|Merged||
|Remove pyflakes from CI|[#4371](https://github.com/onnx/onnx/pull/4371)|Merged||
|Print model name in the exception message|[#4378](https://github.com/onnx/onnx/pull/4378)|Merged||
|Upgrade to `macos-11` due to `macos-10.15` deprecation|[#4379](https://github.com/onnx/onnx/pull/4379)|Merged||
|Install ONNX before linting|[#4428](https://github.com/onnx/onnx/pull/4428)|Merged||
|Fix Windows Release Python 3.10 CI failure when building NumPy from source|[#4440](https://github.com/onnx/onnx/pull/4440)|Merged||
|Fix global LGTM Python/C++ warnings and fully enable it|[#4467](https://github.com/onnx/onnx/pull/4467)|Merged||
|Upgrade macOS in azp due to upcoming deprecation|[#4471](https://github.com/onnx/onnx/pull/4471)|Merged||
|Install specified mypy version first in Python Lint CI to make mypy version consistent|[#4498](https://github.com/onnx/onnx/pull/4498)|Merged||
|Validate the generated number of `input.pb` and `output.pb` and new node files in CIs|[#4514](https://github.com/onnx/onnx/pull/4514)|Merged||
|Use GitHub code scanning before retiring deprecated LGTM.com|[#4524](https://github.com/onnx/onnx/pull/4524)|Merged||
|Add flake8-bugbear in checks|[#4557](https://github.com/onnx/onnx/pull/4557)|Merged||
|Editable_mode=False for now to enable pip install -e . with latest setuptools|[#4558](https://github.com/onnx/onnx/pull/4558)|Merged||
|Build onnx in page creating CI|[#4570](https://github.com/onnx/onnx/pull/4570)|Merged||
|Implement type and shape inference for ArrayFeatureExtractor|[#4572](https://github.com/onnx/onnx/pull/4572)|Merged||
|Manually catch build succeeded for page creating CI|[#4576](https://github.com/onnx/onnx/pull/4576)|Merged||
|Only manually trigger page deployment and remove duplicate style check in Linux-CI|[#4597](https://github.com/onnx/onnx/pull/4597)|Merged||
|Fix LayerNormalization equation|[#4607](https://github.com/onnx/onnx/pull/4607)|Merged||
|Upgrade to newer Protobuf 3.20.2|[#4629](https://github.com/onnx/onnx/pull/4629)|Merged||
|revert AttributeHasValue|[#4630](https://github.com/onnx/onnx/pull/4630)|Merged||
|Use Python 3.10 in CI job CodeQL|[#4633](https://github.com/onnx/onnx/pull/4633)|Merged||
|Fix enforce-style CI error: solve empty-body issue raised by mypy 0.990|[#4643](https://github.com/onnx/onnx/pull/4643)|Merged||
|Fix _parse_repo_info|[#4648](https://github.com/onnx/onnx/pull/4648)|Merged||
|Fix weekly mac release CI: only submit source distribution in one CI|[#4656](https://github.com/onnx/onnx/pull/4656)|Merged||
|Correct types in GridSample def.cc|[#4655](https://github.com/onnx/onnx/pull/4655)|Merged||

### Documentation updates
| Description | PR | Status | Notes |      
|--------------------------------------|:------:|:------------:|:------------:|
|Fix some typos|[#4228](https://github.com/onnx/onnx/pull/4228)|Merged||
|Remove prerequisites from installation instruction |[#4247](https://github.com/onnx/onnx/pull/4247)|Merged||
|Update README with build modifications|[#4259](https://github.com/onnx/onnx/pull/4259)|Merged||
|Update `repo_guidelines.md`|[#4306](https://github.com/onnx/onnx/pull/4306)|Merged||
|Merge is needed only there are changes directly added to the release|[#4308](https://github.com/onnx/onnx/pull/4308)|Merged||
|Clarify in the doc that ONNX should use rc as release candidate|[#4333](https://github.com/onnx/onnx/pull/4333)|Merged||
|Update `readme.md`|[#4349](https://github.com/onnx/onnx/pull/4349)|Merged||
|Fix typo in `AddNewOp.md`|[#4353](https://github.com/onnx/onnx/pull/4353)|Merged||
|Add docs for cpp tests requiring built libraries|[#4401](https://github.com/onnx/onnx/pull/4401)|Merged||
|Update PR template|[#4434](https://github.com/onnx/onnx/pull/4434)|Merged||
|Update issues template and ONNX Hub doc|[#4441](https://github.com/onnx/onnx/pull/4441)|Merged||
|Remove sentences about deprecated `onnx-docker` in docs|[#4455](https://github.com/onnx/onnx/pull/4455)|Merged||
|Create feature request form|[#4479](https://github.com/onnx/onnx/pull/4479)|Merged||
|Add `vcpkg` instruct step|[#4531](https://github.com/onnx/onnx/pull/4531)|Merged||
|Fix documentation generation|[#4564](https://github.com/onnx/onnx/pull/4564)|Merged||
|Format Python code in documents with black|[#4530](https://github.com/onnx/onnx/pull/4530)|Merged||
|Updating based on steering committee discussion on 2022-09-14|[#4573](https://github.com/onnx/onnx/pull/4573)|Merged||
|Update ScatterND documentation|[#4575](https://github.com/onnx/onnx/pull/4575)|Merged||
|fix spelling new python doc|[#4598](https://github.com/onnx/onnx/pull/4598)|Merged||
|Add a link to the HTML documentation from the main page|[#4594](https://github.com/onnx/onnx/pull/4594)|Merged||
|Clarify reshape behavior|[#4601](https://github.com/onnx/onnx/pull/4601)|Merged||
|Update maxpool doc|[#4612](https://github.com/onnx/onnx/pull/4612)|Merged||
|Rename onnx_docs_folder to operator in docs|[#4623](https://github.com/onnx/onnx/pull/4623)|Merged||
|fix some spelling and grammar issues|[#4637](https://github.com/onnx/onnx/pull/4637)|Merged||
|Fix link to package documentation on README.md|[#4649](https://github.com/onnx/onnx/pull/4649)|Merged||
|update docs for onnx.save_model|[#4662](https://github.com/onnx/onnx/pull/4662)|Merged||
|Fixed Typo in concepts.rst|[#4661](https://github.com/onnx/onnx/pull/4661)|Merged||

### Security updates
| Description | PR | Status | Notes |      
|--------------------------------------|:------:|:------------:|:------------:|
|Use fully qualified pathname when loading DLL to prevent security vulnerability|[#4377](https://github.com/onnx/onnx/pull/4377)|Merged||
|Add comment to make sure `vsnprintf` usage is secure|[#4377](https://github.com/onnx/onnx/pull/4377)|Merged||
|Solve vulnerability issue while loading external tensors|[#4508](https://github.com/onnx/onnx/pull/4508)|Merged||

### Deprecation notice
* `TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE` has been deprecated [#4270](https://github.com/onnx/onnx/pull/4270)
* ONNXIFI: ONNX Interface for Framework Integration has been deprecated [#4431](https://github.com/onnx/onnx/pull/4431)

### Partner validation requests
* [onnxruntime](https://github.com/microsoft/onnxruntime/issues/TBD)
* [pytorch](https://github.com/pytorch/pytorch/issues/TBD)
* [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow/issues/TBD)
* [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx/issues/TBD)
* [sklearn-onnx](https://github.com/onnx/sklearn-onnx/issues/TBD)
* [onnxmltools](https://github.com/onnx/onnxmltools/issues/TBD)
* [keras-onnx](https://github.com/onnx/keras-onnx/issues/TBD)
* [onnx-tensort](https://github.com/onnx/onnx-tensorrt/issues/TBD)
* [onnx-coreml](https://github.com/onnx/onnx-coreml/issues/TBD)

# Contributors
TODO: Update contributors at CF
Thanks to these individuals for their contributions in this release since last 1.12.0 release. (Contributor list obtained with: https://github.com/onnx/onnx/graphs/contributors?from=2022-02-08&to=2022-05-24&type=c): @jcwchen, @gramalingam, @xuzijian629, @garymm, @diyessi, @liqunfu, @jantonguirao, @daquexian, @fdwr, @andife, @wschin, @xadupre, @xkszltl, @snnn
