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

import fnmatch
from collections import namedtuple
from contextlib import contextmanager
import os
import hashlib
import shlex
import shutil
import subprocess
import sys
import struct
import tempfile
from textwrap import dedent
import multiprocessing

from tools.ninja_builder import ninja_build_ext
import glob
import json

try:
    import ninja # noqa
    WITH_NINJA = True
    WITH_NINJA = False
except ImportError:
    WITH_NINJA = False

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'onnx')
TP_DIR = os.path.join(TOP_DIR, 'third_party')
PROTOC = find_executable('protoc')
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

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
# Pre Check
################################################################################

assert find_executable('cmake'), 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


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


true_or_die(PROTOC, 'Could not find "protoc" executable! Please install the protobuf compiler, header, and libraries.')


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


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """
    user_options = [
        (str('jobs='), str('j'), str('Specifies the number of jobs to use with make'))
    ]

    built = False

    def initialize_options(self):
        self.jobs = multiprocessing.cpu_count()

    def finalize_options(self):
        self.jobs = int(self.jobs)

    def run(self):
        if cmake_build.built:
            return
        cmake_build.built = True
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [
                find_executable('cmake'),
                '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DPY_VERSION={}'.format('{0}.{1}'.format(*sys.version_info[:2])),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                '-DBUILD_PYTHON=ON'.format(sysconfig.get_python_inc()),
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DONNX_NAMESPACE={}'.format(ONNX_NAMESPACE),
                '-DONNX_USE_MSVC_STATIC_RUNTIME=ON',
            ]
            if os.name == 'nt':
                if 8 * struct.calcsize("P") == 64:
                    # Temp fix for CI
                    # TODO: need a better way to determine generator
                    cmake_args.append('-DCMAKE_GENERATOR_PLATFORM=x64')
            if ONNX_ML:
                cmake_args.append('-DONNX_ML=1')
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            if find_executable('make'):
                build_args = [find_executable('make')]
                # control the number of concurrent jobs
                if self.jobs is not None:
                    build_args.extend(['-j', str(self.jobs)])
            else:
                # Windows CI environment
                build_args = [find_executable('cmake'), '--build', '.']
            subprocess.check_call(build_args)


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        self.run_command('build_proto')
        self.run_command('cmake_build')
        return setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        self.run_command('build_py')
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
    def get_outputs(self):
        return [self.build_lib]

    def run(self):
        self.run_command('build_proto')
        return setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            lib_path = CMAKE_BUILD_DIR
            if os.name == 'nt':
                debug_lib_dir = os.path.join(lib_path, "Debug")
                release_lib_dir = os.path.join(lib_path, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_path = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_path = release_lib_dir
            src = os.path.join(lib_path, filename)

            if not os.path.exists(src):
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), "onnx", filename)
                self.copy_file(src, dst)
                i += 1


cmdclass = {
    'build_proto': build_proto,
    'build_proto_in': build_proto_in,
    'create_version': create_version,
    'cmake_build': cmake_build,
    'build_py': build_py,
    'develop': develop,
    'build_ext': build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(
        name=str(('onnx.' if os.name != 'nt' else '') + 'onnx_cpp2py_export'),
        sources=[])
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
