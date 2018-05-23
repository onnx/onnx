#!/usr/bin/env python

import subprocess
import os


def main():  # type: () -> None
    try:
        root_folder = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.chdir(root_folder)

        subprocess.check_call(["mypy", "."])
        subprocess.check_call(["mypy", "--py2", "."])

        exit(0)
    except subprocess.CalledProcessError:
        # Catch this exception because we don't want it to output a backtrace that would clutter the mypy output
        exit(1)


if __name__ == '__main__':
    main()
