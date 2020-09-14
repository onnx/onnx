import argparse
import sys
import os
import glob
import onnx
import shutil
import tarfile
import re

from onnx import onnx_cpp2py_export

"""
file: get_official_tests.py
description: This script walks through a set of onnx releases fetching the test cases
for each, which contains the model and a set of inputs and expected outputs.
Once the data is fetched, it organizes in folders the test cases according to its domain
and opset version.
"""


def parse_args():  # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser()
    parser.add_argument('--release',
                        help='Set for a specific release. By default all releases are crawled. Must match a release existing on PyPI')
    parser.add_argument('--domain',
                        help='Set to fetch only a specific domain ai.onnx|ai.onnx.ml|ai.onnx.preview.training')
    parser.add_argument('--modelpath', default='test_data/node_official_data', help='Output folder to place the models')
    parser.add_argument('--casepath', default='test_data/node_official_case', help='Output folder to place the scripts')
    parser.add_argument('--getdata', default=False, const=True, nargs='?', help='Downloads the pb test files')
    parser.add_argument('--getcase', default=False, const=True, nargs='?',
                        help='Downloads the Python scripts that generate the models')
    return parser.parse_args()


"""
This table can be accessed programatically (see onnx #2918)
However I suspect there is a bug with the '1.2' entry, since such version
does not exist. Note also that v1.0 and 1.1 doesn't contain a .onnx model
but a .pb one. No model version or opset import is available.
This table must be manually updated if a new onnx version is released.
"""
VERSION_TABLE = [
    # Release-version, IR version, ai.onnx version, ai.onnx.ml version, (optional) ai.onnx.training version
    # ('1.0', 3, 1, 1),
    # ('1.1', 3, 5, 1),
    ('1.1.2', 3, 6, 1),
    ('1.2.3', 3, 7, 1),  # The original table is 1.2 but that release does not exist
    ('1.3', 3, 8, 1),
    ('1.4.1', 4, 9, 1),
    ('1.5.0', 5, 10, 1),
    ('1.6.0', 6, 11, 2),
    ('1.7.0', 7, 12, 2, 1)
]

TEMP_DIR = "temp"

# onnx path for models
PYTHON_TESTS = "onnx/backend/test/data/node/"

# onnx path for python scripts generating the models
PYTHON_CASE = "onnx/backend/test/case/node/"

ALL_SCHEMAS = onnx_cpp2py_export.defs.get_all_schemas_with_history()


def download(onnx_version):
    os.system(f"pip download --no-binary=:all: --no-deps -d {TEMP_DIR} onnx=={onnx_version}")


def copy_files(domain, opset, test_name, test_folder, out_folder):
    # Copy model files by default
    output_path = os.path.join(out_folder,
                               domain,
                               str(opset),
                               test_name)

    # Folder for domain
    domain_path = os.path.join(out_folder, domain)

    # Folder for opset 
    opset_path = os.path.join(out_folder,
                              domain,
                              str(opset))

    os.makedirs(domain_path, exist_ok=True)
    os.makedirs(opset_path, exist_ok=True)

    try:
        shutil.copytree(test_folder, output_path)
    except FileExistsError:
        print("Ignoring", output_path, "already exists")


def get_data(tests_path, domain, out_path):
    # Iterate test cases of a given release
    for test_folder in os.listdir(tests_path):
        test_folder_full = os.path.join(tests_path, test_folder)
        if os.path.isdir(test_folder_full):

            # Check if the test contains a model file. Note that onnx is used
            # in the last releases but .pb is used in the first ones
            included_extensions = ['model.onnx', 'node.pb']
            file_names = [fn for fn in os.listdir(test_folder_full) if
                          any(fn.endswith(ext) for ext in included_extensions)]

            if len(file_names) != 1:
                raise Exception("Model file not found")

            # Open the onnx file
            onnx_path = os.path.join(tests_path, test_folder, file_names[0])
            onnx_model = onnx.load(onnx_path)

            if len(onnx_model.opset_import) != 1:
                raise Exception("Opset imported different than one")

            # Check which opset is imported
            opset_version = onnx_model.opset_import[0].version
            domain_name = onnx_model.opset_import[0].domain

            # Base domain ai.onnx is empty
            if domain_name == "":
                domain_name = "ai.onnx"

            # Just skip if a specific domain is provided
            if domain and domain != domain_name:
                continue

            copy_files(domain_name, opset_version, test_folder,
                       test_folder_full, out_path)

def get_node(node_path, domain, out_path, version):
    for test in os.listdir(node_path):
        if test != "__init__.py":
            case_full_path = os.path.join(node_path, test)

            case_f = open(case_full_path, 'r')
            content = case_f.read().replace("\n", "").strip().replace(" ", "")
            aux1 = content.split("make_node('")[1::2]
            operator = [i.split("',")[0] for i in aux1]

            if not operator:
                print("Skipping file", case_full_path)
                continue
            else:
                domain_name = [i.domain for i in ALL_SCHEMAS if i.name == operator[0]]

            if not domain_name:
                domain_name = "custom"
                print("Operator", operator, "was not found, so it should be a custom")
            else:
                domain_name = domain_name[0]

            # Base domain ai.onnx is empty
            if domain_name == "":
                domain_name = "ai.onnx"

            # Just skip if a specific domain is provided
            if domain and domain != domain_name:
                continue

            mappings = {"ai.onnx": version[2],
                        "ai.onnx.ml": version[3],
                        "ai.onnx.preview.training": version[4] if len(version) > 4 else '-1',
                        # this field is optional
                        "custom": 1}

            os.makedirs(os.path.join(out_path, domain_name), exist_ok=True)
            os.makedirs(os.path.join(out_path, domain_name, str(mappings[domain_name])), exist_ok=True)
            shutil.copy(case_full_path, os.path.join(out_path, domain_name, str(mappings[domain_name])))


def main():  # type: () -> None
    args = parse_args()

    if not args.getdata and not args.getcase:
        raise Exception("You must set either getdata or getcase")

    # If no release is provided, crawl the whole table
    if not args.release:
        releases = VERSION_TABLE
    else:
        # Only allow specific releases that we know its opset_version mapping
        releases = [version for version in VERSION_TABLE if version[0] == args.release]
        if not releases:
            raise Exception("The provided release is not in the table")

    # Iterate the PyPI releases
    for version in releases:
        release = version[0]

        # Download the Python release
        download(release)

        # Untar the files
        tar_file_path = glob.glob(f'{TEMP_DIR}/onnx-{release}*.tar.gz')[0]
        folder_name = tar_file_path.replace(".tar.gz", "")
        onnx_tar_release = tarfile.open(tar_file_path, mode='r')
        onnx_tar_release.extractall(f"{TEMP_DIR}")

        # Path to the tests folder containing all the onnx and pb files
        tests_path = os.path.join(folder_name, PYTHON_TESTS)
        node_path = os.path.join(folder_name, PYTHON_CASE)

        if args.getdata:
            get_data(tests_path, args.domain, args.modelpath)
            print("Done processing data for release", release)

        # if getcase is enabled, copy also the python scripts that generate the models
        if args.getcase:
            get_node(node_path, args.domain, args.casepath, version)
            print("Done processing scripts for release", release)

    os.system(f"rm -r {TEMP_DIR}")


if __name__ == '__main__':
    main()