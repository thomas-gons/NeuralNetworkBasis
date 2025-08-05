#include "layer.hpp"


DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = xt::random::rand({output_size, input_size}, -1.0f, 1.0f);
    biases = xt::zeros<float>({output_size}); 
}

DenseLayer::DenseLayer(const xt::xarray<float>& weights, const xt::xarray<float>& biases) :
    weights(weights), biases(biases) {}

xt::xarray<float> DenseLayer::forward(const xt::xarray<float>& inputs) {
    this->inputs = inputs;
    return xt::linalg::dot(inputs, xt::transpose(weights)) + biases; 
}

xt::xarray<float> DenseLayer::backward(const xt::xarray<float>& upstream_gradient, float lr) {
    size_t batch_size = inputs.shape()[0];
    
    auto weights_gradient = xt::linalg::dot(xt::transpose(inputs), upstream_gradient);
    auto biases_gradient = xt::sum(upstream_gradient, {0});

    weights -= lr * xt::transpose(weights_gradient) / batch_size; 
    biases -= lr * biases_gradient / batch_size;

    return xt::linalg::dot(upstream_gradient, weights);
}

float SigmoidLayer::activation_function(float weighted_sum) {
    return 1.0 / (1.0 + std::exp(-weighted_sum));
}

xt::xarray<float> SigmoidLayer::forward(const xt::xarray<float>& inputs) {
    auto vecf = xt::vectorize([this](float x) {
        return this->activation_function(x);
    });
    last_output = vecf(inputs);
    return last_output;
}

xt::xarray<float> SigmoidLayer::backward(const xt::xarray<float>& upstream_gradient, float lr) {
    return upstream_gradient * last_output * (1 - last_output);
}


float ReLULayer::activation_function(float weighted_sum) {
    return std::max((float)0.0f, weighted_sum);
}

xt::xarray<float> ReLULayer::forward(const xt::xarray<float>& inputs) {
    auto vecf = xt::vectorize([this](float x) {
        return this->activation_function(x);
    });

    this->inputs = inputs;
    last_output = vecf(inputs);
    return last_output;
}

xt::xarray<float> ReLULayer::backward(const xt::xarray<float>& upstream_gradient, float lr) {
    return upstream_gradient * xt::cast<float>(this->inputs > 0);
}
