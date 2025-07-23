#include "layer.hpp"



DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(Eigen::MatrixXd::Random(output_size, input_size)),
      biases(Eigen::VectorXd::Random(output_size)) {}


Eigen::VectorXd DenseLayer::forward(Eigen::VectorXd& inputs) {
        return weights * inputs + biases;
}

Eigen::VectorXd DenseLayer::backward(Eigen::VectorXd& upstream_gradient, double lr) {
    weights *= -lr * upstream_gradient * inputs;
    biases *= -lr * upstream_gradient;
}

ActivationLayer::ActivationLayer() = default;
ActivationLayer::~ActivationLayer() = default;

Eigen::VectorXd ActivationLayer::forward(Eigen::VectorXd& inputs) {
    last_output = inputs.unaryExpr(activation_function);
    return last_output;
}

SigmoidLayer::SigmoidLayer() = default;
SigmoidLayer::~SigmoidLayer() = default;

double SigmoidLayer::activation_function(double weighted_sum) {
    return 1 / (1 + std::exp(-weighted_sum));
}

Eigen::VectorXd SigmoidLayer::backward(Eigen::VectorXd& upstream_gradient, double lr) {
    auto ones = Eigen::VectorXd::Ones(last_output.size());
    return last_output * (ones - last_output) * upstream_gradient;
}