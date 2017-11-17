### ONNX Backend Test

#### What is ONNX Backend Test

ONNX Backend Test is a test suite that each ONNX backend should run to verify whether it fulfills ONNX's standard. It serves both as a verification tool for backend implementations and one of the two ways to define each operator's expected behavior (the other way is to add it to the documentation).

There are two types of tests in this suite – Node Tests and Model Tests:

- **Node Tests** verify whether a backend is performing the correct computation, having the expected behavior of handling various attributes for each individual operator. In each test case, the backend will be given a node with some input, and the returned output will be compared with an expected output.
- **Model Tests** verify the backend at the model level. The test cases are similar to those of Node Tests', but instead of a node, the backend will be given an ONNX model.

#### Contributing

As ONNX aims to become the spec of deep learning models format, it's important to ensure that there is no ambiguity in each ONNX operator's definition; adding more test cases is the only way to enforce this.

Node Tests are created as Python/Numpy code in https://github.com/onnx/onnx/tree/master/onnx/backend/test/case/node, and then exported to protobuf files to https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node as the source of truth by invoking the shell command `backend-test-tools generate-data`. Test cases of each operator lives in one standalone file, e.g. for the operator [Add](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add), its test cases are in [add.py](https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/add.py), and each `expect(...)` statement in the code corresponds to one test case. The source code of all `export.*` functions will be also embedded as example code snippets in the [Operators documentation page](https://github.com/onnx/onnx/blob/master/docs/Operators.md). You are contributing to both the test and the documentation!

For Model Tests, since each model protobuf file can be large in size, we don't place the file directly in the repo. Rather, we upload them to the cloud, and download them on demand when running the tests. Each test case consists of one model definition protobuf file, and several pairs of input and output files. Adding a new test case involves some manual work from admins (like uploading the files to the cloud), so if you have an ONNX model that you would like to contribute, please contact us.
