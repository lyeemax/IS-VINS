#ifndef CS_VINS_LINEAR9_FACTOR_H
#define CS_VINS_LINEAR9_FACTOR_H
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

using namespace Eigen;
class Linear9Factor : public ceres::SizedCostFunction<9, 9> {
public:
    Linear9Factor() = delete;

    Linear9Factor(const Matrix<double,9,1> VB_,ceres::Ownership ownership=ceres::DO_NOT_TAKE_OWNERSHIP):VB(VB_) {
        linearized=false;
        jacobians.resize(1);
        residual.resize(9,1);
    }
    void setIndex(int i) {
        index=i;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Vi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Vector3d Bai(parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d Bgi(parameters[0][6], parameters[0][7], parameters[0][8]);

        Eigen::Matrix<double,9,1> VB_;
        VB_<<Vi,Bai,Bgi;
        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);

        residual=VB_-VB;
        residual=sqrt_info*residual;

//        double error=residual.transpose()*residual;
//        if(error>0.01) {
//            cerr << "residual of VB prior of " << index << " +++: " << error << endl;
//        }
        if (jacobians){
            if (jacobians[0]) {
                if(linearized){
                    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i=this->jacobians[0];
                    jacobian_pose_i=sqrt_info*jacobian_pose_i;
                }else{
                    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setIdentity();
                    jacobian_pose_i=sqrt_info*jacobian_pose_i;
                }

            }

        }

        return true;
    }

    void EvaluateOnlyJacobians(double const *VBi,bool Nscheck=false){
        Eigen::Vector3d Vi(VBi[0], VBi[1], VBi[2]);
        Eigen::Vector3d Bai(VBi[3], VBi[4], VBi[5]);
        Eigen::Vector3d Bgi(VBi[6], VBi[7], VBi[8]);
        Eigen::Matrix<double,9,1> VB_;
        VB_<<Vi,Bai,Bgi;
        residual=VB_-VB;
        Matrix<double,9,9> I99;
        I99.setIdentity();

        jacobians[0]=I99;
        if(!Nscheck){
            linearized=0;
        }

    }
    Matrix<double,9,1> VB;
    MatrixXd sqrt_info;
    vector<MatrixXd> jacobians;
    int index;
    bool linearized;
    VectorXd residual;
};
#endif //CS_VINS_LINEAR9_FACTOR_H
