#include "layer.hpp"

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = xt::random::rand({output_size, input_size}, -1.0f, 1.0f);
    biases = xt::zeros<double>({output_size}); 
}

xt::xarray<double> DenseLayer::forward(const xt::xarray<double>& inputs) {
    this->inputs = inputs;
    return xt::linalg::dot(inputs, xt::transpose(weights)) + biases; 
}

xt::xarray<double> DenseLayer::backward(const xt::xarray<double>& upstream_gradient, double lr) {
    // std::cout << "BACKWARD: dense layer" << std::endl;
    // std::cout << "upstream gradient: " << upstream_gradient << std::endl;
    auto weights_gradient = xt::linalg::outer(inputs, upstream_gradient);
    weights -= lr * xt::transpose(weights_gradient);
    biases -= lr * upstream_gradient;
    return xt::linalg::dot(upstream_gradient, weights);
}

double SigmoidLayer::activation_function(double weighted_sum) {
    return 1.0 / (1.0 + std::exp(-weighted_sum));
}

xt::xarray<double> SigmoidLayer::forward(const xt::xarray<double>& inputs) {
    auto vecf = xt::vectorize([this](double x) {
        return this->activation_function(x);
    });

    last_output = vecf(inputs);
    return last_output;
}

xt::xarray<double> SigmoidLayer::backward(const xt::xarray<double>& upstream_gradient, double lr) {
    return upstream_gradient * last_output * (1 - last_output);
}


double ReLULayer::activation_function(double weighted_sum) {
    return std::max((double)0.0f, weighted_sum);
}

xt::xarray<double> ReLULayer::forward(const xt::xarray<double>& inputs) {
    auto vecf = xt::vectorize([this](double x) {
        return this->activation_function(x);
    });

    this->inputs = inputs;
    last_output = vecf(inputs);
    return last_output;
}

xt::xarray<double> ReLULayer::backward(const xt::xarray<double>& upstream_gradient, double lr) {
    return upstream_gradient * xt::cast<double>(this->inputs > 0);
}
