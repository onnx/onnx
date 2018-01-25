===========================
Generating the Build System
===========================

Prerequisites:
- Tools
  - CMake: https://cmake.org/download/
  - Git
  - Visual Studio (2015)

Steps:
1. Create a sub folder "build" under cmake folder.
mkdir build
2. Go to folder "build".
cd build
3. Call cmake to generate solution.
cmake .. -A x64 -T v140
4. Open onnxir.sln.
onnxir.sln
5. Build onnxir.