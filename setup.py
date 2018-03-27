from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, dep_util, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

import platform
import fnmatch
from collections import namedtuple
import os
import hashlib
import shutil
import subprocess
import sys
import tempfile
from textwrap import dedent

from tools.ninja_builder import ninja_build_ext
import glob
import json

try:
    import ninja # noqa
    WITH_NINJA = True
except ImportError:
    WITH_NINJA = False

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx')
TP_DIR = os.path.join(TOP_DIR, 'third_party')
PROTOC = find_executable('protoc')

DEFAULT_ONNX_NAMESPACE = 'onnx'
ONNX_ML = bool(os.getenv('ONNX_ML') == '1')
ONNX_NAMESPACE = os.getenv('ONNX_NAMESPACE', DEFAULT_ONNX_NAMESPACE)

install_requires = ['six']
setup_requires = []
tests_require = []

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )

################################################################################
# Utilities
################################################################################


def die(msg):
    log.error(msg)
    sys.exit(1)


def true_or_die(b, msg):
    if not b:
        die(msg)
    return b


def recursive_glob(directory, pattern):
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(directory)
            for f in fnmatch.filter(files, pattern)]

# https://stackoverflow.com/a/3431838/2143581


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

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
        else:
            print("Warning: Environment Variable PROTOBUF_INCDIR or CONDA_PREFIX is not set, which may cause protobuf including folder error.")

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
        tmp_dir = tempfile.mkdtemp()
        gen_script = os.path.join(SRC_DIR, 'gen_proto.py')
        stems = ['onnx', 'onnx-operators']

        in_files = [gen_script]
        out_files = []
        need_rename = (ONNX_NAMESPACE != DEFAULT_ONNX_NAMESPACE)
        for stem in stems:
            in_files.append(
                os.path.join(SRC_DIR, '{}.in.proto'.format(stem)))
            if ONNX_ML:
                proto_base = '{}_{}-ml'.format(stem,
                                               ONNX_NAMESPACE) if need_rename else '{}-ml'.format(stem)
                if need_rename:
                    out_files.append(os.path.join(SRC_DIR, '{}-ml.pb.h'.format(stem)))
            else:
                proto_base = '{}_{}'.format(stem, ONNX_NAMESPACE) if need_rename else stem
                if need_rename:
                    out_files.append(os.path.join(SRC_DIR, '{}.pb.h'.format(stem)))
            out_files.extend([
                os.path.join(SRC_DIR, '{}_pb.py'.format(stem.replace('-', '_'))),
                os.path.join(SRC_DIR, '{}.proto'.format(proto_base)),
                os.path.join(SRC_DIR, '{}.proto3'.format(proto_base)),
            ])

        log.info('compiling *.in.proto to temp dir {}'.format(tmp_dir))
        command_list = [
            sys.executable, gen_script,
            '-p', ONNX_NAMESPACE,
            '-o', tmp_dir
        ]
        if ONNX_ML:
            command_list.append('--ml')
        subprocess.check_call(command_list + stems)

        for out_f in out_files:
            tmp_f = os.path.join(tmp_dir, os.path.basename(out_f))
            if os.path.exists(out_f) and md5(out_f) == md5(tmp_f):
                log.info("Skip updating {} since it's the same.".format(out_f))
                continue
            log.info("Copying {} to {}".format(tmp_f, out_f))
            shutil.copyfile(tmp_f, out_f)
        shutil.rmtree(tmp_dir)


class build_proto(ONNXCommand):
    def run(self):
        self.run_command('build_proto_in')

        stems = ['onnx', 'onnx-operators']
        need_rename = (ONNX_NAMESPACE != DEFAULT_ONNX_NAMESPACE)
        for stem in stems:
            if ONNX_ML:
                proto_base = '{}_{}-ml'.format(stem,
                                               ONNX_NAMESPACE) if need_rename else '{}-ml'.format(stem)
            else:
                proto_base = '{}_{}'.format(stem, ONNX_NAMESPACE) if need_rename else stem

            proto = os.path.join(SRC_DIR, '{}.proto'.format(proto_base))
            pb2 = "{}_{}".format(stem.replace('-', '_'), ONNX_NAMESPACE.replace('-',
                                                                                '_')) if need_rename else stem.replace('-', '_')
            if ONNX_ML:
                pb2 += "_ml"
            outputs = [
                os.path.join(SRC_DIR, '{}.pb.cc'.format(proto_base)),
                os.path.join(SRC_DIR, '{}.pb.h'.format(proto_base)),
                os.path.join(SRC_DIR, '{}_pb2.py'.format(pb2)),
                os.path.join(SRC_DIR, '{}_pb.py'.format(stem.replace('-', '_'))),
            ]
            if ONNX_ML:
                outputs.append(os.path.join(SRC_DIR, '{}-ml.pb.h'.format(stem)))
            else:
                outputs.append(os.path.join(SRC_DIR, '{}.pb.h'.format(stem)))
            if self.force or any(dep_util.newer(proto, o) for o in outputs):
                log.info('compiling {}'.format(proto))
                subprocess.check_call([
                    PROTOC,
                    '--proto_path', SRC_DIR,
                    '--python_out', SRC_DIR,
                    '--cpp_out', SRC_DIR,
                    proto
                ])


class create_version(ONNXCommand):
    def run(self):
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(dedent('''\
            # This file is generated by setup.py. DO NOT EDIT!

            from __future__ import absolute_import
            from __future__ import division
            from __future__ import print_function
            from __future__ import unicode_literals

            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        self.run_command('build_proto')
        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        setuptools.command.develop.develop.run(self)
        self.create_compile_commands()

    def create_compile_commands(self):
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        ninja_files = glob.glob('build/*_compile_commands.json')
        all_commands = [entry for f in ninja_files for entry in load(f)]
        with open('compile_commands.json', 'w') as f:
            json.dump(all_commands, f, indent=2)


build_ext_parent = ninja_build_ext if WITH_NINJA \
    else setuptools.command.build_ext.build_ext


class build_ext(build_ext_parent):
    def run(self):
        self.run_command('build_proto')
        for ext in self.extensions:
            ext.pre_run()
        return setuptools.command.build_ext.build_ext.run(self)


cmdclass = {
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


def create_extension(ExtType, name, sources, dependencies, extra_link_args, extra_objects, define_macros):
    include_dirs = sum([dep.include_dirs for dep in dependencies], [TOP_DIR])
    libraries = sum([dep.libraries for dep in dependencies], [])
    extra_compile_args = ['-std=c++11']
    if sys.platform == 'darwin':
        extra_compile_args.append('-stdlib=libc++')
    if os.getenv('CONDA_PREFIX'):
        include_dirs.append(os.path.join(os.getenv('CONDA_PREFIX'), "include"))
    if platform.system() == 'Windows':
        extra_compile_args.append('/MT')
    return ExtType(
        name=name,
        define_macros=define_macros,
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
        need_rename = (ONNX_NAMESPACE != DEFAULT_ONNX_NAMESPACE)

        original_onnx = [
            os.path.join(SRC_DIR, "onnx.pb.cc"),
            os.path.join(SRC_DIR, "onnx-operators.pb.cc"),
        ]
        original_onnx_ml = [
            os.path.join(SRC_DIR, "onnx-ml.pb.cc"),
            os.path.join(SRC_DIR, "onnx-operators-ml.pb.cc"),
        ]
        if ONNX_ML:
            # Remove onnx.pb.cc, onnx-operators.pb.cc from sources.
            sources_filter = original_onnx
            if need_rename:
                sources_filter.extend(original_onnx_ml)
        else:
            # Remove onnx-ml.pb.cc, onnx-operators-ml.pb.cc from sources.
            sources_filter = original_onnx_ml
            if need_rename:
                sources_filter.extend(original_onnx)

        for source_filter in sources_filter:
            if source_filter in self.sources:
                self.sources.remove(source_filter)


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

define_macros = [('ONNX_NAMESPACE', ONNX_NAMESPACE)]
if ONNX_ML:
    define_macros.append(('ONNX_ML', '1'))

ext_modules = [
    create_extension(ONNXCpp2PyExtension,
                     str('onnx.onnx_cpp2py_export'),
                     sources=[],  # sources will be propagated in pre_run
                     dependencies=cpp2py_deps,
                     extra_link_args=cpp2py_link_args,
                     extra_objects=cpp2py_extra_objects,
                     define_macros=define_macros)
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
packages = setuptools.find_packages()

install_requires.extend(['protobuf', 'numpy'])

################################################################################
# Test
################################################################################

setup_requires.append('pytest-runner')
tests_require.append('pytest-cov')
tests_require.append('nbval')
tests_require.append('tabulate')

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
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    author='bddppq',
    author_email='jbai@fb.com',
    url='https://github.com/onnx/onnx',
    entry_points={
        'console_scripts': [
            'check-model = onnx.bin.checker:check_model',
            'check-node = onnx.bin.checker:check_node',
            'backend-test-tools = onnx.backend.test.cmd_tools:main',
        ]
    },
)
