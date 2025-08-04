#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "common.hpp"

class Layer {
public:
    xt::xarray<double> inputs;
    xt::xarray<double> last_output;

    virtual ~Layer() = default;

    virtual xt::xarray<double> forward(const xt::xarray<double>& inputs) = 0;
    virtual xt::xarray<double> backward(const xt::xarray<double>& upstream_gradient, double lr) = 0;
};

class DenseLayer: public Layer {
public:
    xt::xarray<double> weights;
    xt::xarray<double> biases;

public:
    DenseLayer(int input_size, int output_size);

    xt::xarray<double> forward(const xt::xarray<double>& inputs) override;
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradient, double lr) override;
};

class ActivationLayer: public Layer {
public:
    virtual double activation_function(double weighted_sum) = 0;
    virtual xt::xarray<double> forward(const xt::xarray<double>& inputs) = 0;
    virtual xt::xarray<double> backward(const xt::xarray<double>& upstream_gradient, double lr) = 0;
};

class SigmoidLayer: public ActivationLayer {
public:
    SigmoidLayer() = default;
    ~SigmoidLayer() override = default;
    
    double activation_function(double weighted_sum) override;
    xt::xarray<double> forward(const xt::xarray<double>& inputs) override;
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradient, double lr) override;
};

class ReLULayer: public ActivationLayer {
public:
    ReLULayer() = default;
    ~ReLULayer() override = default;
    
    double activation_function(double weighted_sum) override;
    xt::xarray<double> forward(const xt::xarray<double>& inputs) override;
    xt::xarray<double> backward(const xt::xarray<double>& upstream_gradient, double lr) override;
};

#endif