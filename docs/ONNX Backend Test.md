### ONNX Backend Test

#### What is ONNX Backend Test

ONNX Backend Test is a test suite that each ONNX Backend should run to verify whether it fullfils ONNX's standard. It serves both as a verification tool for backend implemtations and one of the two ways to define each operator's expected behavior (the other way is to write it down in the documentation).

There are two types of tests in this suite: Node Tests and Model Tests.

- Node Tests is intended to verify whether a backend is doing the correct computation, having the expected bahavior of handling varies attributes for each individual operator. In each test case, backend will be given a node and some inputs, its returned outputs will then be compared with the expected outputs.
- Model Tests is intended to verify the backend at the model level, test cases are similar as in the Node tests, but instead of a node, the backend will be given an ONNX model.

#### Contributing

As ONNX aims to become the spec of Deep Learning models format, it's super important to make sure there is no ambiguity in each ONNX operator's definition, and adding more test cases is the only way to enforce this.

Node Tests are created as Python/Numpy code in https://github.com/onnx/onnx/tree/master/onnx/backend/test/case/node, and then exported to protobuf files to https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node as the source of the truth by invoking the shell command `backend-test-tools generate-data`. Test cases of each operator lives in one standalone file, e.g. for operator [Add](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add), its test cases are in [add.py](https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/add.py), and each `expect(...)` statement in the code corresponds to one test case. The source code of all `export.*` functions will be also embedded as example code snippets in the [Operators documentation page](https://github.com/onnx/onnx/blob/master/docs/Operators.md). You are contibuting to both the test and the documentation!

For Model Tests, since each model can take couple hundred megabytes in size, we don't directly put the model protobuf file in the repo, instead we upload them to the cloud, and download them on demand during running the tests. Each test case consists of one model definition protobuf file, several pairs of inputs and outputs files. Adding new test case involves some manual work from admins (like uploading the files to the cloud) at this point, so if you have an ONNX model that you want to contribute, please contact us.