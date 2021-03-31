#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0


import subprocess
import os


def main():  # type: () -> None
    try:
        root_folder = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.chdir(root_folder)
        # Use --no-site-packages to prevent mypy catching other typecheck errors which are not related to ONNX itself
        subprocess.check_call(["mypy", ".", "--no-site-packages"])
        subprocess.check_call(["mypy", "--py2", ".", "--no-site-packages"])

        exit(0)
    except subprocess.CalledProcessError:
        # Catch this exception because we don't want it to output a backtrace that would clutter the mypy output
        exit(1)


if __name__ == '__main__':
    main()
