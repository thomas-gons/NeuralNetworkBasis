#include "loss.hpp"


float MSELoss::forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
    auto diff = predicted - truth;
    float sum_of_squares = xt::sum(xt::square(diff))();
    size_t batch_size = predicted.shape()[0];
    return sum_of_squares / (2 * batch_size);
}

xt::xarray<float> MSELoss::backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
    return predicted - truth;
}

