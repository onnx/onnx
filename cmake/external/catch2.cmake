# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
FetchContent_Declare(
  Catch2
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.10.0.tar.gz
)
FetchContent_MakeAvailable(Catch2)
