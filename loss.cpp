#include "loss.hpp"


namespace loss {

    float MSE::forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        auto diff = predicted - truth;
        float sum_of_squares = xt::sum(xt::square(diff))();
        size_t batch_size = predicted.shape()[0];
        return sum_of_squares / (2 * batch_size);
    }

    xt::xarray<float> MSE::backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        return predicted - truth;
    }

    float CrossEntropy::forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        auto log = xt::log(predicted);
        auto sum = xt::sum(truth * log)();
        return -sum;
    }

    xt::xarray<float> CrossEntropy::backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        return -truth / predicted;
    }

    xt::xarray<float> CrossEntropy::backward_fused(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        return predicted - truth;
    }
}