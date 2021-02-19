
import argparse
#import glob
import os
import re
import shutil
import subprocess
import sys
import platform

#cmake_args = '-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON -DONNX_USE_LITE_PROTO=ON'
#os.environ['CMAKE_ARGS'] = cmake_args

def is_windows():
    return True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNXRuntime CI build driver.",
        usage="""  # noqa
        Default behavior is --update_submodules --build --test.

        The Update phase will update git submodules.
        The Build phase will run cmake to generate makefiles and build all projects.
        The Test phase will run all tests including pytest and c++ tests.

        Use the individual flags to only run the specified stages.

        In order to build protobuf one needs to specify build_protobuf, protobuf_build_dir and protobuf_install dirs
        """)
    # Main arguments
    parser.add_argument("--config", default="Debug", choices=["Debug", "Release"], help="Configuration to build.")
    parser.add_argument("--build_dir", required=False, help="Path to the build directory.", default=os.getcwd())

    parser.add_argument("--build_protobuf", action='store_false', help="Builds protobuf with same cmake generator and config as ONNX")
    parser.add_argument("--protobuf_build_dir", required=False, help="Path to the protobuf build directory.")
    parser.add_argument("--protobuf_install_dir", required=False, help="Path to the protobuf install directory.")

    parser.add_argument("--update_submodules", action='store_true', help="Update submodules")
    parser.add_argument("--build", action='store_true', help="Build")
    parser.add_argument("--test", action='store_true', help="Test")
    parser.add_argument("--skip_tests", action='store_true', help="Skip all tests")
    parser.add_argument("--test_python", action='store_true', help="Test")
    parser.add_argument("--test_cplusplus", action='store_true', help="Test")

    parser.add_argument("--use_msvc_static_runtime", default=True, help="When set to true link to CRT statically")
    parser.add_argument("--use_protobuf_shared_libs", default=False, help="Whether to use protobuf shared libsor static libs.")
    parser.add_argument("--use_lite_proto", default=True, help="Use protobuf lite")
    parser.add_argument("--disable_werror", action='store_true', help="disables treatment of warnings as error")
    parser.add_argument("--cmake_generator", choices=['Visual Studio 15 2017', 'Visual Studio 16 2019'], 
        default='Visual Studio 16 2019' if is_windows() else None,
        help="Specify the generator that CMake invokes. This is only supported on Windows")

    return parser.parse_args()

def run_subprocess(args, cwd=None, capture_stdout=False, dll_path=None,
                   shell=False, env={}):
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    my_env.update(env)

    return subprocess.check_call(args, cwd=cwd, shell=shell, env=my_env)

def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"],
                   cwd=source_dir)

def set_env(args):
    if args.config == 'Debug':
        os.environ['DEBUG'] = '1'


def set_cmake_extra_defines(args):
    cmake_args = "-DONNX_BUILD_TESTS=ON -DONNX_ML=1"

    if is_windows():
        cmake_args += ' -G ' + args.cmake_generator

    if args.use_msvc_static_runtime:
        cmake_args += ' -DONNX_USE_MSVC_STATIC_RUNTIME=ON'

    if args.use_lite_proto:
        cmake_args +=' -DONNX_USE_LITE_PROTO=ON'
    else:
        cmake_args += ' -DONNX_USE_LITE_PROTO=OFF'

    if args.disable_werror:
        cmake_args +=' -DONNX_WERROR=OFF'
    else:
        cmake_args += ' -DONNX_WERROR=ON'

    if args.use_protobuf_shared_libs:
        cmake_args += ' -DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DProtobuf_USE_STATIC_LIBS=OFF'
    else:
        cmake_args += ' -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON'


    os.environ['CMAKE_ARGS'] = cmake_args

def build_protobuf(args):
    if args.protobuf_build_dir == "" or args.protobuf_install_dir == "":
        raise Exception("Protobuf build dir and install dir are required when --build_protobuf option is selected.")

    build_type = 'Release'
    if args.config == 'Debug':
        build_type = 'Debug'

    # create cmake args for protobuf    
    cmake_args = [
        CMAKE,
        '-DCMAKE_BUILD_TYPE=%s' % build_type,
        '-DCMAKE_INSTALL_PREFIX={}'.format(args.protobuf_install_dir),
        '-Dprotobuf_BUILD_EXAMPLES=OFF',
        '-Dprotobuf_BUILD_TESTS=OFF'
    ]

    #if is_windows():
        #cmake_args.extend(['-G ', args.cmake_generator])

    if platform.architecture()[0] == '64bit':
        cmake_args.extend(['-A', 'x64', '-T', 'host=x64'])
    else:
        cmake_args.extend(['-A', 'Win32', '-T', 'host=x86'])

    if args.use_msvc_static_runtime:
        cmake_args.extend(['-Dprotobuf_MSVC_STATIC_RUNTIME=ON', '-DProtobuf_USE_STATIC_LIBS=ON'])
    else:
        cmake_args.extend(['-Dprotobuf_MSVC_STATIC_RUNTIME=OFF', '-DProtobuf_USE_STATIC_LIBS=OFF', '-DPROTOBUF_USE_DLLS=1'])



def build_onnx(args):
    if args.update_submodules:
        source_dir = args.build_dir
        update_submodules(source_dir)

    if args.use_msvc_static_runtime and args.use_protobuf_shared_libs:
        print("Both use_msvc_static_runtime and use_protobuf_shared_libs flags are set to true. This is not a supported combination.")
    set_env(args)
    set_cmake_extra_defines(args)

    args = [sys.executable, os.path.join(source_dir, 'setup.py'), 'develop']
    run_subprocess(args, cwd=source_dir)
        
def main():
    args = parse_arguments()

    # If there was no explicit argument saying what to do, default
    # to update, build and test (for native builds).
    if not (args.update_submodules or args.build or args.test):
        print(
            "Defaulting to running update, build "
            "[and test].")
        args.update_submodules = True
        args.build = True
        args.test = True

    if args.skip_tests:
        args.test = False

    if args.test:
        args.test_python = True
        args.test_cplusplus = True

    if args.build:
        build_onnx(args)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(str(e))
        sys.exit(1)

