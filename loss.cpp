#include "loss.hpp"


double MSELoss::forward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) {
    return 0.5 * (predicted - truth).squaredNorm();
}

Eigen::VectorXd MSELoss::backward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) {
    return predicted - truth;
}