# Development

You will need an install of protobuf and numpy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

During development it's convenient to install ONNX in development mode (Note: Set environment variable `ONNX_ML=1` for onnx-ml):

```
git clone --recursive https://github.com/onnx/onnx.git
pip install -e onnx/
```
Then, after you have made changes to

- Python files, the changes are immediately effective in your installation, you do not need to install again.
- C++ files, you need to do install again to trigger the native extension build.

## Folder Structure

- onnx/: the main folder that all code lies under
  - onnx.proto: the protobuf (v2.6.1) that contains all the structures
  - checker.py: utility to check whether a serialized ONNX proto is legal.
  - helper.py: tools for graph operation
  - defs/: subfolder that defines the ONNX operators.
  - test/: test files

## Generated operator documentation

[Operator docs in Operators.md](docs/Operators.md) are auto-generated based on C++ operator definitions. In order to refresh them run the following command from the repo root and commit the results:

```
python onnx/defs/gen_doc.py
```

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, first you need to install pytest:

```
pip install pytest-cov nbval
```

After installing pytest, do

```
pytest
```

to run tests.

# Other developer documentation

* [How to implement ONNX backend (ONNX to something converter)](docs/Implementing%20an%20ONNX%20backend.md)
* [Backend test infrastructure and how to add tests](docs/ONNX%20Backend%20Test.md)

# License

[MIT License](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
