from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

import platform
import fnmatch
from collections import namedtuple
import os
import subprocess
import sys
from textwrap import dedent

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx')
TP_DIR = os.path.join(TOP_DIR, 'third_party')
PROTOC = find_executable('protoc')

install_requires = {'six'}
setup_requires = set()
test_requires = set()

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except subprocess.CalledProcessError:
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )

################################################################################
# Utilities
################################################################################

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def die(msg):
    log(msg)
    sys.exit(1)


def true_or_die(b, msg):
    if not b:
        die(msg)
    return b


def recursive_glob(directory, pattern):
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(directory)
            for f in fnmatch.filter(files, pattern)]

################################################################################
# Pre Check
################################################################################

true_or_die(PROTOC, 'Could not find "protoc" executable!')

################################################################################
# Dependencies
################################################################################

class Dependency(object):
    def __init__(self):
        self.include_dirs = []
        self.libraries = []


class Python(Dependency):
    def __init__(self):
        super(Python, self).__init__()
        self.include_dirs = [sysconfig.get_python_inc()]


class Protobuf(Dependency):
    def __init__(self):
        super(Protobuf, self).__init__()
        # TODO: allow user specify protobuf include_dirs libraries with flags
        use_conda = os.getenv('CONDA_PREFIX') and platform.system() == 'Windows'

        libs = []
        if os.getenv('PROTOBUF_LIBDIR'):
            libs.append(os.path.join(os.getenv('PROTOBUF_LIBDIR'), "libprotobuf"))
        elif use_conda:
            libs.append(os.path.join(os.getenv('CONDA_PREFIX'), "Library", "lib", "libprotobuf"))
        else:
            libs.append("protobuf")

        includes = []
        if os.getenv('PROTOBUF_INCDIR'):
            includes.append(os.path.join(os.getenv('PROTOBUF_INCDIR')))
        elif use_conda:
            includes.append(os.path.join(os.getenv('CONDA_PREFIX'), "Library", "Include"))

        self.libraries = libs
        self.include_dirs = includes

class Pybind11(Dependency):
    def __init__(self):
        super(Pybind11, self).__init__()
        self.include_dirs = [os.path.join(TP_DIR, 'pybind11', 'include')]

################################################################################
# Customized commands
################################################################################

class ONNXCommand(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class build_proto_in(ONNXCommand):
    def run(self):
        log('compiling onnx.in.proto')
        subprocess.check_call(["python", os.path.join(SRC_DIR, "gen_proto.py")])


class build_proto(ONNXCommand):
    def run(self):
        self.run_command('build_proto_in')

        # NB: Not a glob, because you can't build both onnx.proto and
        # onnx-proto.ml in the same build
        proto_files = [os.path.join(SRC_DIR, "onnx.proto")]

        for proto_file in proto_files:
            log('compiling {}'.format(proto_file))
            if not self.dry_run:
                subprocess.check_call([
                    PROTOC,
                    '--proto_path', SRC_DIR,
                    '--python_out', SRC_DIR,
                    '--cpp_out', SRC_DIR,
                    proto_file
                ])


class create_version(ONNXCommand):
    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        self.run_command('build_proto')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('build_proto')
        for ext in self.extensions:
            ext.pre_run()
        setuptools.command.build_ext.build_ext.run(self)


cmdclass={
    'build_proto': build_proto,
    'build_proto_in': build_proto_in,
    'create_version': create_version,
    'build_py': build_py,
    'develop': develop,
    'build_ext': build_ext,
}

################################################################################
# Extensions
################################################################################

class ONNXExtension(setuptools.Extension):
    def pre_run(self):
        pass

def create_extension(ExtType, name, sources, dependencies, extra_link_args, extra_objects):
    include_dirs = sum([dep.include_dirs for dep in dependencies], [TOP_DIR])
    libraries = sum([dep.libraries for dep in dependencies], [])
    extra_compile_args=['-std=c++11']
    if sys.platform == 'darwin':
        extra_compile_args.append('-stdlib=libc++')
    if os.getenv('CONDA_PREFIX'):
        include_dirs.append(os.path.join(os.getenv('CONDA_PREFIX'), "include"))
    if platform.system() == 'Windows':
        extra_compile_args.append('/MT')
    return ExtType(
        name=name,
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_objects=extra_objects,
        extra_link_args=extra_link_args,
        language='c++',
    )

class ONNXCpp2PyExtension(setuptools.Extension):
    def pre_run(self):
        self.sources = recursive_glob(SRC_DIR, '*.cc')
        if os.path.join(SRC_DIR, "onnx-ml.pb.cc") in self.sources:
            raise RuntimeError("Stale onnx/onnx-ml.pb.cc file detected.  Please delete this file and rebuild.")

cpp2py_deps = [Pybind11(), Python()]
cpp2py_link_args = []
cpp2py_extra_objects = []
build_for_release = os.getenv('ONNX_BINARY_BUILD')
if build_for_release and platform.system() == 'Linux':
    # Cribbed from PyTorch
    # get path of libstdc++ and link manually.
    # for reasons unknown, -static-libstdc++ doesn't fully link some symbols
    CXXNAME = os.getenv('CXX', 'g++')
    path = subprocess.check_output([CXXNAME, '-print-file-name=libstdc++.a'])
    path = path[:-1]
    if type(path) != str:  # python 3
        path = path.decode(sys.stdout.encoding)
    cpp2py_link_args += [path]

    # Hard coded look for the static libraries from Conda
    assert os.getenv('CONDA_PREFIX')
    cpp2py_extra_objects.extend([os.path.join(os.getenv('CONDA_PREFIX'), 'lib', 'libprotobuf.a'),
                             os.path.join(os.getenv('CONDA_PREFIX'), 'lib', 'libprotobuf-lite.a')])
else:
    cpp2py_deps.append(Protobuf())

ext_modules = [
    create_extension(ONNXCpp2PyExtension,
                     str('onnx.onnx_cpp2py_export'),
                     sources=[], # sources will be propagated in pre_run
                     dependencies=cpp2py_deps,
                     extra_link_args=cpp2py_link_args,
                     extra_objects=cpp2py_extra_objects)
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
packages = setuptools.find_packages()

install_requires.update(['protobuf', 'numpy'])

################################################################################
# Test
################################################################################

setup_requires.add('pytest-runner')
test_requires.add('pytest-cov')
test_requires.add('nbval')

################################################################################
# Final
################################################################################

setuptools.setup(
    name="onnx",
    version=VersionInfo.version,
    description="Open Neural Network Exchange",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=test_requires,
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx',
    entry_points={
        'console_scripts': [
            'check-model = onnx.bin.checker:check_model',
            'check-node = onnx.bin.checker:check_node',
        ]
    },
)
