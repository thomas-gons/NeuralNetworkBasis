#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <cassert>

typedef Eigen::Tensor<double, 1> Vector;
typedef Eigen::Tensor<double, 2> Matrix;
typedef Eigen::Tensor<double, Eigen::Dynamic> Tensor;
#endif