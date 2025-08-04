#ifndef __LOSS_HPP__
#define __LOSS_HPP__


#include "common.hpp"

class Loss {
public:
    virtual float forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) = 0;
    virtual xt::xarray<float> backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) = 0;
};

class MSELoss: public Loss {
public:
    MSELoss() = default;
    float forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
    xt::xarray<float> backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
};

#endif