#pragma once

// This file contains backports of STL features for newer C++.

/*
 * Use MOVE_CAPTURE_IF_CPP14 in a lambda capture so it gets
 * copied in C++11 and moved in C++14.
 * Example:
 *   std::string mystring;
 *   auto lambda = [MOVE_CAPTURE_IF_CPP14(mystring)] {
 *     std::cout << mystring;
 *   }
 */
#ifdef __cpp_init_captures
  #define MOVE_CAPTURE_IF_CPP14(variable) variable = std::move(variable)
#else
  #define MOVE_CAPTURE_IF_CPP14(variable) variable
#endif

// make_unique is a C++14 feature. If we don't have 14, we will emulate
// its behavior. This is copied from folly/Memory.h
#if __cplusplus < 201402L &&                                               \
    !defined __cpp_lib_make_unique &&                                      \
    (!defined(_MSC_VER) || _MSC_VER < 1900)

namespace std {

template<typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template<typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template<typename T, typename... Args>
typename std::enable_if<
  std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;

}

#endif
