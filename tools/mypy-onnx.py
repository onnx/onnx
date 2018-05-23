#!/usr/bin/env python

import subprocess
import os


def main():  # type: () -> None
    try:
        root_folder = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        source_folder = os.path.join(root_folder, "onnx/test")
        os.chdir(root_folder)

        subprocess.check_call(["mypy", "."])
        subprocess.check_call(["mypy", "--py2", "."])

        # Since test cases aren't a python package (missing __init__.py),
        # mypy ignores them. Explicitly call mypy with these files.

        # Enumerate py and pyi files, they're listed without file extension
        py_files = [os.path.relpath(os.path.join(dirpath, f), root_folder)
                   for dirpath, dirnames, files in os.walk(source_folder)
                   for f in files if f.endswith('.py')]

        subprocess.check_call(["mypy"] + py_files)
        subprocess.check_call(["mypy", "--py2"] + py_files)

        exit(0)
    except subprocess.CalledProcessError:
        # Catch this exception because we don't want it to output a backtrace that would clutter the mypy output
        exit(1)


if __name__ == '__main__':
    main()
