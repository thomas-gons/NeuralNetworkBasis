#ifndef __LOSS_HPP__
#define __LOSS_HPP__


#include "common.hpp"

class Loss {
public:
    virtual double forward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) = 0;
    virtual xt::xarray<double> backward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) = 0;
};

class MSELoss: public Loss {
public:
    MSELoss() = default;
    double forward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) override;
    xt::xarray<double> backward(const xt::xarray<double>& predicted, const xt::xarray<double>& truth) override;
};

#endif