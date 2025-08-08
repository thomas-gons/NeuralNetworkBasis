#ifndef __LOSS_HPP__
#define __LOSS_HPP__


#include "common.hpp"

namespace loss {
    class Loss {
    public:
        virtual float forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) = 0;
        virtual xt::xarray<float> backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) = 0;
    };

    class MSE: public Loss {
    public:
        MSE() = default;
        float forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
        xt::xarray<float> backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
    };

    class CrossEntropy: public Loss {
    public:
        CrossEntropy() = default;
        float forward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
        xt::xarray<float> backward(const xt::xarray<float>& predicted, const xt::xarray<float>& truth) override;
        xt::xarray<float> backward_fused(const xt::xarray<float>& predicted, const xt::xarray<float>& truth);
    };
}

#endif