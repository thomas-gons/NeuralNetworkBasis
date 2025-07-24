#include "loss.hpp"

double MSELoss::forward(const Tensor& predicted, const Tensor& truth) {
    return 0.5 * (predicted - truth).squaredNorm();
}

Tensor MSELoss::backward(const Tensor& predicted, const Tensor& truth) {
    return predicted - truth;
}
