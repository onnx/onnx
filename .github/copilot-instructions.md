We use lintrunner as the linter:

```sh
# Display all lints and apply the fixes
lintrunner -a --output oneline
# Or apply fixes only (faster)
lintrunner f --output oneline
```

To build ONNX:

```sh
python -m pip install --quiet --upgrade pip setuptools wheel
export ONNX_BUILD_TESTS=0
export ONNX_ML=1
python -m pip install .
```
