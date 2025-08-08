#include "layer.hpp"


DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = xt::random::rand({output_size, input_size}, -1.0f, 1.0f);
    biases = xt::zeros<float>({output_size}); 
}

DenseLayer::DenseLayer(const xt::xarray<float>& weights, const xt::xarray<float>& biases) :
    weights(weights), biases(biases) {}

xt::xarray<float> DenseLayer::forward(const xt::xarray<float>& inputs) {
    this->inputs = inputs;
    auto transposed_weights = xt::transpose(weights);
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

namespace activation {

    float Sigmoid::activation_function(float weighted_sum) {
        return 1.0 / (1.0 + std::exp(-weighted_sum));
    }

    xt::xarray<float> Sigmoid::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        return upstream_gradient * last_output * (1 - last_output);
    }

    
    float Tanh::activation_function(float weighted_sum) {
        return std::tanh(weighted_sum);
    }

    xt::xarray<float> Tanh::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        return upstream_gradient * (1 - xt::square(last_output));
    }


    float ReLU::activation_function(float weighted_sum) {
        return std::max((float)0.0f, weighted_sum);
    }

    xt::xarray<float> ReLU::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        return upstream_gradient * xt::cast<float>(this->inputs > 0);
    }


    float LeakyReLU::activation_function(float weighted_sum) {
        return weighted_sum >= 0 ? weighted_sum : 0.01f * weighted_sum;
    }

    xt::xarray<float> LeakyReLU::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        auto grad_mask = xt::cast<float>(this->inputs > 0);
        return upstream_gradient * (grad_mask + 0.01f * (1 - grad_mask));
    }


    float ELU::activation_function(float weighted_sum) {
        return weighted_sum >= 0 ? weighted_sum : alpha * (std::exp(weighted_sum) - 1);
    }

    xt::xarray<float> ELU::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        auto mask = last_output >= 0;
        auto derivative = xt::where(mask, 1.0f, alpha * xt::exp(last_output));
        return upstream_gradient * derivative;
    }

    float GELU::activation_function(float weighted_sum) {
        return 0.5 * weighted_sum * (1 + std::tanh(std::sqrt(2 / M_PI) * (weighted_sum + 0.044715 * std::pow(weighted_sum, 3))));
    }

    xt::xarray<float> GELU::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        auto tanh_val = xt::tanh(sqrt_2_over_pi * (last_output + 0.044715f * xt::pow(last_output, 3)));
        auto derivative = 0.5f * (1.0f + tanh_val + last_output * (1 - xt::pow(tanh_val, 2)) * sqrt_2_over_pi * (1.0f + 0.134145f * xt::pow(last_output, 2)));
        return upstream_gradient * derivative;
    }


    xt::xarray<float> Softmax::forward(const xt::xarray<float>& inputs) {
        xt::xarray<float> exp_inputs = xt::exp(inputs);
        float sum_exp = xt::sum(exp_inputs)();

        this->inputs = inputs;
        last_output = exp_inputs / sum_exp;
        return last_output; 
    }

    xt::xarray<float> Softmax::backward(const xt::xarray<float>& upstream_gradient, float lr) {
        size_t batch_size = last_output.shape()[0];
        size_t num_classes = last_output.shape()[1];

        auto y = last_output;

        auto y_col = xt::expand_dims(y, 2);
        auto y_row = xt::expand_dims(y, 1);
        xt::xarray<float> yyT = y_col * y_row;

        auto I_batched = xt::eye(num_classes).reshape({1, num_classes, num_classes});
        auto y_reshaped = y.reshape({batch_size, num_classes, 1});
        xt::xarray<float> diag_y = xt::eval(y_reshaped * I_batched);

        xt::xarray<float> J = diag_y - yyT;
        
        xt::xarray<float> gradients_to_z = xt::xarray<float>::from_shape({batch_size, num_classes});
        for (size_t i = 0; i < batch_size; ++i) {
            auto J_sample = xt::view(J, i, xt::all(), xt::all());
            auto up_grad_sample = xt::view(upstream_gradient, i, xt::all());
            xt::view(gradients_to_z, i, xt::all()) = xt::linalg::dot(J_sample, up_grad_sample);
        }
        
        return gradients_to_z;
    }

}