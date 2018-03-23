#pragma once

// This file contains backports of STL features for newer C++.

#ifdef __cpp_init_captures
  #define MOVE_CAPTURE_IF_CPP14(variable) variable = std::move(variable)
#else
  #define MOVE_CAPTURE_IF_CPP14(variable) variable
#endif
