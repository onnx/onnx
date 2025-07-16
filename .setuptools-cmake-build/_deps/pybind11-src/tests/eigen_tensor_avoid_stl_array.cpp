/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#ifndef EIGEN_AVOID_STL_ARRAY
#    define EIGEN_AVOID_STL_ARRAY
#endif

#include "test_eigen_tensor.inl"

PYBIND11_MODULE(eigen_tensor_avoid_stl_array, m, pybind11::mod_gil_not_used()) {
    eigen_tensor_test::test_module(m);
}
