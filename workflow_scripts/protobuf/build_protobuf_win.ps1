# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter()]
    [String]$arch = "x64",

    [Parameter()]
    [String]$build_type = "Release"
)

echo "Build protobuf from source on Windows."
Invoke-WebRequest -Uri https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz -OutFile protobuf.tar.gz -Verbose
tar -xvf protobuf.tar.gz
cd protobuf-3.21.12
$protobuf_root_dir = Get-Location
mkdir protobuf_install
cd cmake
cmake -G "Visual Studio 17 2022" -A $arch -DCMAKE_INSTALL_PREFIX="../protobuf_install" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=$build_type
msbuild INSTALL.vcxproj /p:Configuration=$build_type
echo "Protobuf installation complete."
echo "Set paths"
$protoc_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\bin"
$protoc_lib_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\lib"
$protobuf_include_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\include"
$Env:PATH="$protoc_path;$protoc_lib_path;$protobuf_include_path;$ENV:PATH"
$($Env:PATH).Split(';')
protoc
cd ../../
