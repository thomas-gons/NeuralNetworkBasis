#ifndef __LOSS_HPP__
#define __LOSS_HPP__

#include <Eigen/Dense>
#include <cmath>

class Loss {
public:
    virtual ~Loss() = default;

    virtual double forward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) = 0;
    virtual Eigen::VectorXd backward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) = 0;
};

class MSELoss: public Loss {
    double forward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) override;

    Eigen::VectorXd backward(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& truth) override;
};

#endif