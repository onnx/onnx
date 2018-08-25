#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import os
import re
from textwrap import dedent

autogen_header = """\
//
// WARNING: This file is automatically generated!  Please edit onnx.in.proto.
//


"""

LITE_OPTION = '''
option optimize_for = LITE_RUNTIME;

'''

DEFAULT_PACKAGE_NAME = "onnx"

IF_ONNX_ML_REGEX = re.compile(r'\s*//\s*#if\s+ONNX-ML\s*$')
ENDIF_ONNX_ML_REGEX = re.compile(r'\s*//\s*#endif\s*$')
ELSE_ONNX_ML_REGEX = re.compile(r'\s*//\s*#else\s*$')


MYPY = False
if MYPY:
    from typing import Iterable, Text


def process_ifs(lines, onnx_ml):  # type: (Iterable[Text], bool) -> Iterable[Text]
    in_if = 0
    for line in lines:
        if IF_ONNX_ML_REGEX.match(line):
            assert 0 == in_if
            in_if = 1
        elif ELSE_ONNX_ML_REGEX.match(line):
            assert 1 == in_if
            in_if = 2
        elif ENDIF_ONNX_ML_REGEX.match(line):
            assert (1 == in_if or 2 == in_if)
            in_if = 0
        else:
            if 0 == in_if:
                yield line
            elif (1 == in_if and onnx_ml):
                yield line
            elif (2 == in_if and not onnx_ml):
                yield line


IMPORT_REGEX = re.compile(r'(\s*)import\s*"([^"]*)\.proto";\s*$')
PACKAGE_NAME_REGEX = re.compile(r'\{PACKAGE_NAME\}')
ML_REGEX = re.compile(r'(.*)\-ml')


def process_package_name(lines, package_name):  # type: (Iterable[Text], Text) -> Iterable[Text]
    need_rename = (package_name != DEFAULT_PACKAGE_NAME)
    for line in lines:
        m = IMPORT_REGEX.match(line) if need_rename else None
        if m:
            include_name = m.group(2)
            ml = ML_REGEX.match(include_name)
            if ml:
                include_name = "{}_{}-ml".format(ml.group(1), package_name)
            else:
                include_name = "{}_{}".format(include_name, package_name)
            yield m.group(1) + 'import "{}.proto";'.format(include_name)
        else:
            yield PACKAGE_NAME_REGEX.sub(package_name, line)


PROTO_SYNTAX_REGEX = re.compile(r'(\s*)syntax\s*=\s*"proto2"\s*;\s*$')
OPTIONAL_REGEX = re.compile(r'(\s*)optional\s(.*)$')


def convert_to_proto3(lines):  # type: (Iterable[Text]) -> Iterable[Text]
    for line in lines:
        # Set the syntax specifier
        m = PROTO_SYNTAX_REGEX.match(line)
        if m:
            yield m.group(1) + 'syntax = "proto3";'
            continue

        # Remove optional keywords
        m = OPTIONAL_REGEX.match(line)
        if m:
            yield m.group(1) + m.group(2)
            continue

        # Rewrite import
        m = IMPORT_REGEX.match(line)
        if m:
            yield m.group(1) + 'import "{}.proto3";'.format(m.group(2))
            continue

        yield line


def translate(source, proto, onnx_ml, package_name):  # type: (Text, int, bool, Text) -> Text
    lines = source.splitlines()  # type: Iterable[Text]
    lines = process_ifs(lines, onnx_ml=onnx_ml)
    lines = process_package_name(lines, package_name=package_name)
    if proto == 3:
        lines = convert_to_proto3(lines)
    else:
        assert proto == 2
    return "\n".join(lines)  # TODO: not Windows friendly


def qualify(f, pardir=os.path.realpath(os.path.dirname(__file__))):  # type: (Text, Text) -> Text
    return os.path.join(pardir, f)


def convert(stem, package_name, output, do_onnx_ml=False, lite=False):  # type: (Text, Text, Text, bool) -> None
    proto_in = qualify("{}.in.proto".format(stem))
    need_rename = (package_name != DEFAULT_PACKAGE_NAME)
    if do_onnx_ml:
        proto_base = "{}_{}-ml".format(stem, package_name) if need_rename else "{}-ml".format(stem)
    else:
        proto_base = "{}_{}".format(stem, package_name) if need_rename else "{}".format(stem)
    proto = qualify("{}.proto".format(proto_base), pardir=output)
    proto3 = qualify("{}.proto3".format(proto_base), pardir=output)

    print("Processing {}".format(proto_in))
    with io.open(proto_in, 'r') as fin:
        source = fin.read()
        print("Writing {}".format(proto))
        with io.open(proto, 'w', newline='') as fout:
            fout.write(autogen_header)
            fout.write(translate(source, proto=2, onnx_ml=do_onnx_ml, package_name=package_name))
            if lite:
                fout.write(LITE_OPTION)
        print("Writing {}".format(proto3))
        with io.open(proto3, 'w', newline='') as fout:
            fout.write(autogen_header)
            fout.write(translate(source, proto=3, onnx_ml=do_onnx_ml, package_name=package_name))
            if lite:
                fout.write(LITE_OPTION)
        if need_rename:
            if do_onnx_ml:
                proto_header = qualify("{}-ml.pb.h".format(stem), pardir=output)
            else:
                proto_header = qualify("{}.pb.h".format(stem), pardir=output)
            print("Writing {}".format(proto_header))
            with io.open(proto_header, 'w', newline='') as fout:
                fout.write("#pragma once\n")
                fout.write("#include \"{}.pb.h\"\n".format(proto_base))

    # Generate py mapping
    # "-" is invalid in python module name, replaces '-' with '_'
    pb_py = qualify('{}_pb.py'.format(stem.replace('-', '_')), pardir=output)
    if need_rename:
        pb2_py = qualify('{}_pb2.py'.format(proto_base.replace('-', '_')), pardir=output)
    else:
        if do_onnx_ml:
            pb2_py = qualify('{}_ml_pb2.py'.format(stem.replace('-', '_')), pardir=output)
        else:
            pb2_py = qualify('{}_pb2.py'.format(stem.replace('-', '_')), pardir=output)

    print('generating {}'.format(pb_py))
    with open(pb_py, 'w') as f:
        f.write(str(dedent('''\
        # This file is generated by setup.py. DO NOT EDIT!

        from __future__ import absolute_import
        from __future__ import division
        from __future__ import print_function
        from __future__ import unicode_literals

        from .{} import *  # noqa
        '''.format(os.path.splitext(os.path.basename(pb2_py))[0]))))


def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='Generates .proto file variations from .in.proto')
    parser.add_argument('-p', '--package', default='onnx',
                        help='package name in the generated proto files'
                        ' (default: %(default)s)')
    parser.add_argument('-m', '--ml', action='store_true', help='ML mode')
    parser.add_argument('-l', '--lite', action='store_true',
                        help='generate lite proto to use with protobuf-lite')
    parser.add_argument('-o', '--output',
                        default=os.path.realpath(os.path.dirname(__file__)),
                        help='output directory (default: %(default)s)')
    parser.add_argument('stems', nargs='*', default=['onnx', 'onnx-operators'],
                        help='list of .in.proto file stems '
                        '(default: %(default)s)')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for stem in args.stems:
        convert(stem,
                package_name=args.package,
                output=args.output,
                do_onnx_ml=args.ml,
                lite=args.lite)


if __name__ == '__main__':
    main()
