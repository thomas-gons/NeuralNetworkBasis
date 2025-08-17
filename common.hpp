#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/core/xvectorize.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <nlohmann/json.hpp>

std::string xarray_shape(const xt::xarray<float>& arr);
nlohmann::json load_json(const std::string& path);

#endif