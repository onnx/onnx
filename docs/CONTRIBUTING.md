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

## Adding a new operator

For implementors in the ONNX community to be able to effectively implement any new operators proposed, pull requests should include:

- Operator description - just 1 to 2 sentences describing the operation (e.g. “The new Expand operator broadcasts an input tensor to an output shape using a shape tensor”). A more complete description will be found in the actual PR, but readers should be able to get a summary up front without reading through the diff.
- Justification of usefulness - what's the motivation and value in adding this operator into ONNX? (e.g. “This is supported by XYZ frameworks already and is seeing broad usage across the community of speech analysis.”)
- Links to existing examples and usage, where possible:
  - Example model/algorithm that uses the operation? (e.g. WaveNet, Detectron, etc.)
  - What frameworks already have this operation? (e.g. CNTK, Tensorflow, etc.)
  - Which research papers describe or reference it? (e.g. arxiv paper)
- Sample operator implementation in Python, with minimal dependencies except numpy.

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

# Static typing (mypy)

We use [mypy](http://mypy-lang.org/) to run static type checks on the onnx code base. To check that your code passes, you'll first need to install the mypy type checker. If you're using python 3, call from your onnx source folder:

```
pip install -e .[mypy]
```

The type checker cannot run in a python 2 environment (but it will check python 2 code).
If you're using python 2, you need to install mypy into your system packages instead:

```
pip3 install mypy==[version]
```
*Note: You'll find the version we're currently using in `setup.py`.*

After having installed mypy, you can run the type checks:

```
python setup.py typecheck
```


# Other developer documentation

* [How to implement ONNX backend (ONNX to something converter)](ImplementingAnOnnxBackend.md)
* [Backend test infrastructure and how to add tests](OnnxBackendTest.md)

# License

[MIT License](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
