#include "layer.hpp"
#include <iostream>

DenseLayer::DenseLayer(int p_input_dim, int p_output_dim)
    : input_dim(p_input_dim), output_dim(p_output_dim) {
        
        weights(Eigen::array<Eigen::Index, 2>{(Eigen::Index)output_dim, (Eigen::Index)input_dim});
        biases(Eigen::array<Eigen::Index, 1>{(Eigen::Index)output_dim});

        weights.setRandom();
        biases.setConstant(0.0f);
      }

Tensor DenseLayer::forward(Tensor& p_inputs) {
    assert(p_inputs.dimensions().size() == 1 && "Upstream gradient must be a 1D vector (rank 1).");
    assert(p_inputs.dimensions()[0] == output_dim && "Input size must match layer's output_dim.");
    inputs = p_inputs;
    last_output = weights * p_inputs + biases;
    return last_output;
}

Tensor DenseLayer::backward(Tensor& upstream_gradient, double lr) {
    assert(upstream_gradient.dimensions().size() == 1 && "Upstream gradient must be a 1D vector (rank 1).");
    assert(upstream_gradient.dimensions()[0] == output_dim && "Upstream gradient size must match layer's output_dim.");

    weights -= lr * (upstream_gradient * inputs);

    biases -= lr * upstream_gradient;
    Eigen::array<Eigen::Index, 2> input_as_col_dims = {(Eigen::Index)input_dim};
    Eigen::array<Eigen::Index, 2> grad_as_row_dims = {1, (Eigen::Index)output_dim};

    return weights.reshape(weights.dimensions()[1], weights.dimensions()[0]) * upstream_gradient;
}

ActivationLayer::ActivationLayer() = default;
ActivationLayer::~ActivationLayer() = default;

SigmoidLayer::SigmoidLayer() = default;
SigmoidLayer::~SigmoidLayer() = default;

double SigmoidLayer::activation_function(double weighted_sum) {
    return 1.0 / (1.0 + std::exp(-weighted_sum));
}

Tensor SigmoidLayer::forward(Tensor& inputs_param) {
    inputs = inputs_param;
    last_output = inputs_param.unaryExpr([this](double x) { return activation_function(x); });
    return last_output;
}

Tensor SigmoidLayer::backward(Tensor& upstream_gradient, double lr) {
    auto ones = Tensor::Ones(last_output.size());
    return last_output.cwiseProduct(ones - last_output).cwiseProduct(upstream_gradient);
}