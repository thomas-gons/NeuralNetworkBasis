#include "loss.hpp"


double MSELoss::forward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) {
    auto diff = predicted - truth;
    double sum = xt::sum(xt::square(diff))();
    return sum / predicted.size();
}

xt::xarray<double> MSELoss::backward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) {
    return predicted - truth;
}

