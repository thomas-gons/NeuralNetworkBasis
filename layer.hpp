#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include "common.hpp"


class Layer {
public:
    xt::xarray<float> inputs;
    xt::xarray<float> last_output;

    virtual ~Layer() = default;

    virtual xt::xarray<float> forward(const xt::xarray<float>& inputs) = 0;
    virtual xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) = 0;
};

class DenseLayer: public Layer {
public:
    xt::xarray<float> weights;
    xt::xarray<float> biases;

public:
    DenseLayer(int input_size, int output_size);
    DenseLayer(const xt::xarray<float>& weights, const xt::xarray<float>& biases);

    xt::xarray<float> forward(const xt::xarray<float>& inputs) override;
    xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
};

class ActivationLayer: public Layer {
public:
    virtual float activation_function(float weighted_sum) = 0;
    virtual xt::xarray<float> forward(const xt::xarray<float>& inputs) = 0;
    virtual xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) = 0;
};

class SigmoidLayer: public ActivationLayer {
public:
    SigmoidLayer() = default;
    ~SigmoidLayer() override = default;
    
    float activation_function(float weighted_sum) override;
    xt::xarray<float> forward(const xt::xarray<float>& inputs) override;
    xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
};

class ReLULayer: public ActivationLayer {
public:
    ReLULayer() = default;
    ~ReLULayer() override = default;
    
    float activation_function(float weighted_sum) override;
    xt::xarray<float> forward(const xt::xarray<float>& inputs) override;
    xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
};

#endif