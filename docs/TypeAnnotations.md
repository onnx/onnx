<!--- SPDX-License-Identifier: Apache-2.0 -->

# Type annotations for ONNX

At Facebook, we work with community best practices to ensure high code quality, readability and reliability. In line with this, we just added type annotations to our python code to help ONNX developers more easily contribute to the project.

These type annotations are used by [mypy](https://github.com/python/mypy) within the ONNX CI systems to ensure a continuously high code quality standard.
We also have type annotations for our APIs, which means your tools built on top of the ONNX APIs can use static analysis tools like mypy to ensure they are using the APIs correctly.

We expect this to have a positive impact on the reliability of ONNX and dependent projects like ONNX converters - we've also found and fixed a few bugs thanks to the addition of type annotations.

## What is mypy?
Mypy is an opt-in type checker. Annotations are optional and, if present, will be used for static type checking. While it is not advisable to rely too much on mypy to find all errors (for programmers used to a static type system like in C++, it's often confusing that mypy doesn't find some very obvious typing errors), it is still very helpful given the errors it does find.

## Using the type annotations in dependent projects
Say, for example, you're building a converter for ONNX, which converts between ONNX models and the models of some machine learning framework. If you use mypy for your project, it will automatically detect the type hints in the ONNX code and use them to check that you read from and wrote to the ONNX model correctly. Doing this will notify you about identifier typos or wrong data types when accessing attributes of the ONNX protocol buffer or other ONNX features.

## For ONNX contributors
If you're contributing to the ONNX repository, you need to add type annotations to your project or CI will fail. Here is how to install it:

#### Python 3
If you're using Python 3, you can simply install mypy as an extra dependency of onnx.

Call from the top level directory of your onnx repository:

    ~/onnx $ pip install -e .[lint]

And then run the type checks:

    ~/onnx $ python setup.py typecheck

#### Python 2
The mypy type checker needs python 3 to run (even though it can check python 2 code), so if your onnx development environment is set up with python 2, you need to install mypy into your system python 3 packages (i.e. not into your python 2 virtualenv):

    $ pip3 install mypy

Running then works as it does above for python 3.

    ~/onnx $ python setup.py typecheck

Since you've installed mypy manually, you won't get updates for it when we change the version of the mypy dependency in setup.py, so if you're seeing a CI error and can't reproduce it locally, check your mypy version.

### What should I annotate?

Type annotations are (usually) only needed for function arguments and return types, mypy will infer local variable types automatically. "Usually" means there are a few exceptions, but mypy will report these if you hit them.

### What do type annotations look like?
We can't use PEP484 annotations, because we have to support python 2. So we're writing our type annotations in comments. Typed code looks like this:

    def myfunc(arg):  # type: (int) -> bool
      return arg == 2

    class MyClass(object):
      def __init__(self):  # type: () -> None
        pass

      def myfunc(self, arg):  # type: (int) -> bool
        return arg == 2

      # Alternative notation for many parameters
      def myfunc(
        arg1,  # type: int
        arg2,  # type: int
      ):  # type: (...) -> bool
        return arg1 == arg2
