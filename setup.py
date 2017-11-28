from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import distutils
from distutils.spawn import find_executable
from distutils import sysconfig, dep_util, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from contextlib import contextmanager
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

ONNX_ML = bool(os.getenv('ONNX_ML') == '1')

install_requires = ['six']
setup_requires = []
tests_require = []

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
        gen_script = os.path.join(SRC_DIR, 'gen_proto.py')
        stems = ['onnx', 'onnx-operators']

        in_files = [gen_script]
        out_files = []
        for stem in stems:
            in_files.append(
                os.path.join(SRC_DIR, '{}.in.proto'.format(stem)))
            out_files.extend([
                os.path.join(SRC_DIR, '{}.proto'.format(stem)),
                os.path.join(SRC_DIR, '{}.proto3'.format(stem)),
                os.path.join(SRC_DIR, '{}-ml.proto'.format(stem)),
                os.path.join(SRC_DIR, '{}-ml.proto3'.format(stem)),
            ])

        if self.force or any(dep_util.newer_group(in_files, o)
                             for o in out_files):
            log.info('compiling *.in.proto')
            subprocess.check_call([sys.executable, gen_script] + stems)


class build_proto(ONNXCommand):
    def run(self):
        self.run_command('build_proto_in')

        stems = ['onnx', 'onnx-operators']
        for stem in stems:
            if ONNX_ML:
                proto_base = '{}-ml'.format(stem)
            else:
                proto_base = stem

            proto = os.path.join(SRC_DIR, '{}.proto'.format(proto_base))
            # "-" is invalid in python module name, replaces '-' with '_'
            pb_py = os.path.join(SRC_DIR, '{}_pb.py'.format(
                stem.replace('-', '_')))
            pb2_py = os.path.join(SRC_DIR, '{}_pb2.py'.format(
                proto_base.replace('-', '_')))
            outputs = [
                pb_py,
                pb2_py,
                os.path.join(SRC_DIR, '{}.pb.cc'.format(proto_base)),
                os.path.join(SRC_DIR, '{}.pb.h'.format(proto_base)),
            ]
            if self.force or any(dep_util.newer(proto, o) for o in outputs):
                log.info('compiling {}'.format(proto))
                subprocess.check_call([
                    PROTOC,
                    '--proto_path', SRC_DIR,
                    '--python_out', SRC_DIR,
                    '--cpp_out', SRC_DIR,
                    proto
                ])
                log.info('generating {}'.format(pb_py))
                with open(pb_py, 'w') as f:
                    f.write(dedent('''\
                    # This file is generated by setup.py. DO NOT EDIT!

                    from __future__ import absolute_import
                    from __future__ import division
                    from __future__ import print_function
                    from __future__ import unicode_literals

                    from .{} import *  # noqa
                    '''.format(os.path.splitext(os.path.basename(pb2_py))[0])))


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
        return setuptools.command.develop.develop.run(self)


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command('build_proto')
        for ext in self.extensions:
            ext.pre_run()
        return setuptools.command.build_ext.build_ext.run(self)

    def build_extensions(self):
        try:
            import ninja
        except ImportError:
            # Once PEP 518 is in, we can specify ninja as a build time
            # requires. Technically, we can make it work without the
            # PEP by installing ninja to a temp directory and insert
            # that to sys.path, but as at this point ninja integration
            # is not well proven yet, let's just fall back to the
            # default build method.
            return self._build_default()
        else:
            return self._build_with_ninja()

    def _build_default(self):
        return setuptools.command.build_ext.build_ext.build_extensions(self)

    def _build_with_ninja(self):
        import ninja

        build_file = os.path.join(TOP_DIR, 'build.ninja')
        log.debug('Ninja build file at {}'.format(build_file))
        w = ninja.Writer(open(build_file, 'w'))

        w.rule('compile', '$cmd')
        w.rule('link', '$cmd')

        @contextmanager
        def patch(obj, attr_name, val):
            orig_val = getattr(obj, attr_name)
            setattr(obj, attr_name, val)
            yield
            setattr(obj, attr_name, orig_val)

        orig_compile = distutils.unixccompiler.UnixCCompiler._compile
        orig_link = distutils.unixccompiler.UnixCCompiler.link

        def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
            depfile = os.path.splitext(obj)[0] + '.d'
            def spawn(cmd):
                w.build(
                    [obj], 'compile', [src],
                    variables={
                        'cmd': cmd,
                        'depfile': depfile,
                        'deps': 'gcc'
                    })

            extra_postargs.extend(['-MMD', '-MF', depfile])
            with patch(self, 'spawn', spawn):
                orig_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)

        def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):

            w.close()
            ninja._program('ninja', ['-f', build_file])

            orig_link(self, target_desc, objects,
                      output_filename, output_dir, libraries,
                      library_dirs, runtime_library_dirs,
                      export_symbols, debug, extra_preargs,
                      extra_postargs, build_temp, target_lang)

        with patch(distutils.unixccompiler.UnixCCompiler, '_compile', _compile):
            with patch(distutils.unixccompiler.UnixCCompiler, 'link', link):
                with patch(self, 'force', True):
                    self._build_default()

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
    macros = []
    if ONNX_ML:
        macros = [('ONNX_ML', '1')]
    return ExtType(
        name=name,
        define_macros = macros,
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
        if ONNX_ML:
            # Remove onnx.pb.cc, onnx-operators.pb.cc from sources.
            sources_filter = [os.path.join(SRC_DIR, "onnx.pb.cc"), os.path.join(SRC_DIR, "onnx-operators.pb.cc")]
        else:
            # Remove onnx-ml.pb.cc, onnx-operators-ml.pb.cc from sources.
            sources_filter = [os.path.join(SRC_DIR, "onnx-ml.pb.cc"), os.path.join(SRC_DIR, "onnx-operators-ml.pb.cc")]

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
