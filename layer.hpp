#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "common.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(Tensor& inputs) = 0;
    virtual Tensor backward(Tensor& upstream_gradient, double lr) = 0;
};

class DenseLayer: public Layer {
private:
    int input_dim;
    int output_dim;
    Matrix weights;
    Eigen::Tensor<double, 1> biases;

    Eigen::Tensor<double, 1> inputs;
    Eigen::Tensor<double, 1> last_output;

public:
    DenseLayer(int input_size, int output_size);

    Tensor forward(Tensor& inputs) override;
    Tensor backward(Tensor& upstream_gradient, double lr) override;
};

class ActivationLayer: public Layer {
public:
    ActivationLayer();
    virtual ~ActivationLayer();

    virtual double activation_function(double weighted_sum) = 0;
    virtual Tensor forward(Tensor& inputs) = 0;
    virtual Tensor backward(Tensor& upstream_gradient, double lr) = 0;
};

class SigmoidLayer: public ActivationLayer {
public:
    SigmoidLayer();
    ~SigmoidLayer() override;

    double activation_function(double weighted_sum) override;
    Tensor forward(Tensor& inputs) override;
    Tensor backward(Tensor& upstream_gradient, double lr) override;
};

#endif