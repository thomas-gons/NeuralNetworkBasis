#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include "common.hpp"

class Loss {
public:
    virtual ~Loss() = default;

    virtual double forward(const Tensor& predicted, const Tensor& truth) = 0;
    virtual Tensor backward(const Tensor& predicted, const Tensor& truth) = 0;
};

class MSELoss: public Loss {
public:
    double forward(const Tensor& predicted, const Tensor& truth) override;
    Tensor backward(const Tensor& predicted, const Tensor& truth) override;
};

#endif