# Development

__Note:__ If you are building ONNX on Windows, make sure to read the 'Building on Windows' section below before proceeding.

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

## Building on Windows

On Windows, building ONNX requires a little bit more preparation. You should have Visual Studio 2017 installed, and you will need to install [cmake](https://cmake.org/download/): it will be used to generate the Visual Studio projects that are used to actually build ONNX.

The conda-based installation above works well on Windows, but you will need to pay attention to the `PATH` environment variable. Python is installed with Visual Studio 2017, but there are two distributions -- the regular Python distribution, and Anaconda. In each distribution, 'python.exe' and 'pip.exe' are installed in different folders, so there's a risk for inconsistencies.

Before running any command mentioned above, make sure that both the root installation and the Scripts folder are on the `PATH`, and that it is the Anaconda distribution that comes first.

Assuming that Visual Studio was installed in the default location, it should look something like:

```
...;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\Scripts;...
```

Once this is set up consistently, the installation should be smooth.

### Debugging With Visual Studio

Once the developer installation has been run, a Visual Studio solution called `onnx.sln` is available in `.setuptools-cmake-build`, which is a folder created during the installation process. This folder is transient, but useful for development purposes.

Use this solution as you would any other Visual Studio solution -- build, rebuild, clean, etc., but don't count on it sticking around. You can safely remove the entire `.setuptools-cmake-build` folder and redo the `pip install`.

Debugging the C++ code run from Python scripts is straight-forward -- the key is to get the PDB file into the right folder.

To do that, add the following two lines as a post-build event to the `onnx_cpp2py_export` project:

```
copy "$(TargetPath)" "$(SolutionDir)\..\onnx"
copy "$(TargetDir)\onnx_cpp2py_export.pdb" "$(SolutionDir)\..\onnx"
```

Then, use the project property dialog to set the debugging properties of the `onnx_cpp2py_export` project to the following:

|Setting|Value|
|---|---|
|`Command`|`C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe`|
|`Command Arguments`|`onnx\test\checker_test.py`  (or whatever script you want to run)|
|`Working Directory`|the file path to the local ONNX Git repo|

Once this is set up, and `onnx_cpp2py_export` is made the startup project, F5 should just work and any breakpoints set inside the C++ code will be triggered.

Note that it is important that you use the Python interop project to run F5. Otherwise, the build event will not necessarily be triggered and the PDB file won't be updated on each run.

__Note:__ Each time you run `pip install -e .` the project will be regenerated and the build event cleared, so use Visual Studio to update your binaries while you're developing. The debug settings are not affected since they are stored in the `onnx_cpp2py_export.vcxproj.user` file, which cmake doesn't touch.


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
