echo "Install protobuf"
cd ./protobuf_root
$protobuf_root_dir = Get-Location
mkdir protobuf_install
cd ./protobuf/cmake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="../../protobuf_install" -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=Release
msbuild INSTALL.vcxproj /p:Configuration=Release
echo "Protobuf installation complete."
echo "Set paths"
$protoc_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\bin"
$protoc_lib_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\lib"
$protobuf_include_path = Join-Path -Path $protobuf_root_dir -ChildPath "protobuf_install\include"
$Env:PATH="$ENV:PATH;$protoc_path;$protoc_lib_path;$protobuf_include_path"
$($Env:PATH).Split(';')
protoc
