
#ifndef CS_VINS_YAW_FACTOR_H
#define CS_VINS_YAW_FACTOR_H
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <utility/sophus_utils.hpp>
#include "utility/utility.h"
using namespace Eigen;
using namespace std;
class YawFactor : public ceres::SizedCostFunction<1, 7> {
public:
    YawFactor() = delete;

    YawFactor(const Quaterniond Rz) {
        yaw_meas=Rz.inverse()*Vector3d::UnitX();
        jacobians.resize(1);
        residual.resize(1,1);
    }
    void setIndex(int i) {
        index=i;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Map<Eigen::Matrix<double, 1, 1>> residual(residuals);

        Sophus::SO3d Ri(Qi);
        Vector3d res=  Ri*yaw_meas;
        residual=res.block<1,1>(1,0);
        residual=sqrt_info*residual;

        if (jacobians){
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                Matrix<double,3,6> J;
                J.topLeftCorner<3,3>()= Matrix3d::Zero();
                J.bottomRightCorner<3,3>()=-Ri.matrix()*Utility::skewSymmetric(yaw_meas);
                jacobian_pose_i.topLeftCorner<1,6>() =J.block(1,0,1,6);

                jacobian_pose_i=sqrt_info*jacobian_pose_i;
            }
        }
        return true;
    }

    void EvaluateOnlyJacobians(double const *PSi,bool Nscheck=false){
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);

        Sophus::SO3d Ri(Qi);
        Vector3d res=  Ri*yaw_meas;
        residual=res.block<1,1>(1,0);
        Eigen::Matrix<double, 1, 6> jacobian_pose_i;
        jacobian_pose_i.setZero();
        Matrix<double,3,6> J;
        J.topLeftCorner<3,3>()= Matrix3d::Zero();
        J.bottomRightCorner<3,3>()=-Ri.matrix()*Utility::skewSymmetric(yaw_meas);
        jacobian_pose_i.topLeftCorner<1,6>() =J.block(1,0,1,6);
        jacobians[0]=jacobian_pose_i;
    }

    Vector3d yaw_meas;
    MatrixXd sqrt_info;
    int index;
    vector<MatrixXd> jacobians;
    VectorXd residual;
};
#endif //CS_VINS_YAW_FACTOR_H
