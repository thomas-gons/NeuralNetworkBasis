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
        // numerical stability: clip predicted values to avoid log(0) = -inf
        const float epsilon = 1e-15;
        xt::xarray<float> clamped_predicted = xt::clip(predicted, epsilon, 1.0f - epsilon);
        auto log_predicted = xt::log(clamped_predicted);
        auto sum = xt::sum(truth * log_predicted)();
        return -sum;
    }

    xt::xarray<float> CrossEntropy::backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        return -truth / predicted;
    }

    xt::xarray<float> CrossEntropy::backward_fused(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) {
        xt::xarray<float> grad = predicted;

        xt::xarray<size_t> batch_indices = xt::arange(predicted.shape(0));
        for (size_t i = 0; i < batch_indices.size(); ++i) {
            grad(batch_indices(i), truth(i)) -= 1.0f;
        }


        return grad;
    }
}