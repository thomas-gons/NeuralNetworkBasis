#ifndef __LAYER_HPP__
#define __LAYER_HPP__

#include <Eigen/Dense>
#include <cmath>

class Layer {
public:
    Eigen::VectorXd inputs;
    Eigen::VectorXd last_output;

    virtual ~Layer() = default;

    virtual Eigen::VectorXd forward(Eigen::VectorXd& inputs) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd& upstream_gradient, double lr) = 0;
};


class DenseLayer: public Layer {
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;

public:
    DenseLayer(int input_size, int output_size);

    Eigen::VectorXd forward(Eigen::VectorXd& inputs) override;
    Eigen::VectorXd backward(Eigen::VectorXd& upstream_gradient, double lr) override;
};


class ActivationLayer: public Layer {
public:    
    ActivationLayer();
    virtual ~ActivationLayer();

    virtual double activation_function(double weighted_sum) = 0;
    Eigen::VectorXd forward(Eigen::VectorXd& inputs) override;
    virtual Eigen::VectorXd backward(Eigen::VectorXd& upstream_gradient, double lr) = 0;
};


class SigmoidLayer: public ActivationLayer {

public:
    SigmoidLayer();
    ~SigmoidLayer() override;
    
    double activation_function(double weighted_sum) override;
    Eigen::VectorXd backward(Eigen::VectorXd& upstream_gradient, double lr) override;
};

#endif