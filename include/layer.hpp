#ifndef ___HPP__
#define ___HPP__

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

namespace activation {
    class BaseActivation: public Layer {
    public:
        virtual float activation_function(float weighted_sum) {return weighted_sum;};
        xt::xarray<float> forward(const xt::xarray<float>& inputs) {
            auto vecf = xt::vectorize([this](float x) {
                return this->activation_function(x);
            });
            this->inputs = inputs;
            last_output = vecf(inputs);
            return last_output;
        };
        virtual xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) = 0;
    };

    class Sigmoid: public BaseActivation {
    public:
        Sigmoid() = default;
        ~Sigmoid() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class Tanh: public BaseActivation {
    public:
        Tanh() = default;
        ~Tanh() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class ReLU: public BaseActivation {
    public:
        ReLU() = default;
        ~ReLU() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class LeakyReLU: public BaseActivation {
    public:
        LeakyReLU() = default;
        ~LeakyReLU() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class ELU: public BaseActivation {
        float alpha;
    
    public:
        ELU(float alpha) : alpha(alpha) {};
        ~ELU() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class GELU: public BaseActivation {
    public:
        GELU() = default;
        ~GELU() override = default;
        
        float activation_function(float weighted_sum) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };

    class Softmax: public BaseActivation {
    public:
        Softmax() = default;
        ~Softmax() override = default;
        
        xt::xarray<float> forward(const xt::xarray<float>& inputs) override;
        xt::xarray<float> backward(const xt::xarray<float>& upstream_gradient, float lr) override;
    };
}


#endif