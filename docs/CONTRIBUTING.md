# Development

You will need to install protobuf and numpy to build ONNX. An easy
way to get these dependencies is via [Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

During development, it's convenient to install ONNX in development mode (for ONNX-ML, set environment variable `ONNX_ML=1`):

```
git clone --recursive https://github.com/onnx/onnx.git
pip install -e onnx/
```
Then, after you have made changes to Python and C++ files:

- `Python files`: the changes are effective immediately in your installation. You don't need to install these again.
- `C++ files`: you need to install these again to trigger the native extension build.

## Folder structure

- `onnx/`: the main folder that all code lies under
  - `onnx.proto`: the protobuf (v2.6.1) that contains all the structures
  - `checker.py`: a utility to check whether a serialized ONNX proto is legal
  - `helper.py`: tools for graph operation
  - `defs/`: a subfolder that defines the ONNX operators
  - `test/`: test files

## Generated operator documentation

[Operator docs in Operators.md](Operators.md) are automatically generated based on C++ operator definitions. To refresh these docs, remember to re-install (see above) and then run the following command from the repo root and commit the results:

```
python onnx/defs/gen_doc.py
```

# Testing

ONNX uses [pytest](https://docs.pytest.org) as a test driver. To run tests, you'll first need to install pytest:

```
pip install pytest-cov nbval
```

After installing pytest, run

```
pytest
```

to begin the tests.

# Other developer documentation

* [How to implement ONNX backend (ONNX to something converter)](ImplementingAnOnnxBackend.md)
* [Backend test infrastructure and how to add tests](OnnxBackendTest.md)

# License

[MIT License](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
