#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
from typing import Text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(os.path.basename(__file__))
    parser.add_argument('-r', '--root',
                        default=os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                        help='onnx root directory (default: %(default)s)')
    parser.add_argument('-o', '--out', required=True,
                        help='output directory')
    return parser.parse_args()


def gen_trace_file(root_dir: Text, out_path: Text) -> None:
    subprocess.check_output([
        'lcov',
        '-c',
        '-d',
        root_dir,
        '--no-external',
        '--path',
        root_dir,
        '-o',
        out_path])

    subprocess.check_output([
        'lcov',
        '-r',
        out_path,
        os.path.join(root_dir, 'third_party', '*'),
        '-o',
        out_path])

    subprocess.check_output([
        'lcov',
        '-r',
        out_path,
        os.path.join(root_dir, '.setuptools-cmake-build', '*'),
        '-o',
        out_path
    ])


def gen_html_files(root_dir: Text, trace_path: Text, out_dir: Text) -> None:
    subprocess.check_output([
        'genhtml',
        trace_path,
        '-p',
        root_dir,
        '-o',
        out_dir,
    ])


def main() -> None:
    args = parse_args()

    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    if not os.path.exists(out):
        os.makedirs(out)

    trace_path = os.path.join(out, 'onnx-coverage.info')
    gen_trace_file(root, trace_path)

    html_dir = os.path.join(out, 'html')
    gen_html_files(root, trace_path, html_dir)

    print('Static HTML files have been generated at:\n\t{}'.format(html_dir))


if __name__ == '__main__':
    main()
