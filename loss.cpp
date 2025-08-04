#include "loss.hpp"


float MSELoss::forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
    auto diff = predicted - truth;
    float sum = xt::sum(xt::square(diff))();
    return sum / predicted.size();
}

xt::xarray<float> MSELoss::backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
    return predicted - truth;
}

