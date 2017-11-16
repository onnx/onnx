
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

# Development

You will need an install of protobuf and numpy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

During development it's convenient to install ONNX in development mode (Note: Add install option --install-option="--onnxml=1" for onnx-ml):

```
git clone --recursive https://github.com/onnx/onnx.git
pip install -e onnx/
```
Then, after you have made changes to

- Python files, the changes are immediately effective in your installation, you do not need to install again.
- C++ files, you need to do install again to trigger the native extension build.

## Generated operator documentation

[Operator docs in Operators.md](docs/Operators.md) are auto-generated based on C++ operator definitions. In order to refresh them run the following command from the repo root and commit the results:

```
python onnx/defs/gen_doc.py
```


# License

[MIT License](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
