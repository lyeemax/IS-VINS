#ifndef CS_VINS_ROLLPITCH_FACTOR_H
#define CS_VINS_ROLLPITCH_FACTOR_H
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <utility/sophus_utils.hpp>
#include "utility/utility.h"
using namespace Eigen;
using namespace std;
class RollPitchFactor : public ceres::SizedCostFunction<2, 7> {
public:
    RollPitchFactor() = delete;

    RollPitchFactor(const Quaterniond Rz) : R(Rz) {
        jacobians.resize(1);
        residual.resize(2,1);
    }
    void setIndex(int i) {
        index=i;
    }

    void shift(){
        index--;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Map<Eigen::Matrix<double, 2, 1>> residual(residuals);

        Sophus::SO3d Ri(Qi);
        Sophus::SO3d Rmeas(R);
        Vector3d nZ=-1.0*Vector3d::UnitZ();
        Vector3d res= Rmeas * Ri.inverse()*nZ;
        residual=res.head<2>();
        residual=sqrt_info*residual;

        if (jacobians){
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                Matrix<double,3,6> J;
                J.topLeftCorner<3,3>()= Matrix3d::Zero();
                J.bottomRightCorner<3,3>()=Utility::skewSymmetric(Rmeas*Ri.inverse()*nZ)*Rmeas.matrix();
                jacobian_pose_i.topLeftCorner<2,6>() =J.topLeftCorner<2,6>();

                jacobian_pose_i=sqrt_info*jacobian_pose_i;

            }
        }


        //check(parameters,jacobians);
        return true;
    }

    void EvaluateOnlyJacobians(double const *PSi,bool Nscheck=false){
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);

        Sophus::SO3d Ri(Qi);
        Sophus::SO3d Rmeas(R);
        Vector3d nZ=-1.0*Vector3d::UnitZ();
        Vector3d res= Rmeas * Ri.inverse()*nZ;
        residual=res.head<2>();

        Eigen::Matrix<double, 2, 6> jacobian_pose_i;
        jacobian_pose_i.setZero();
        Matrix<double,3,6> J;
        J.topLeftCorner<3,3>()= Matrix3d::Zero();
        J.bottomRightCorner<3,3>()=Utility::skewSymmetric(Rmeas*Ri.inverse()*nZ)*Rmeas.matrix();
        jacobian_pose_i.topLeftCorner<2,6>() =J.topLeftCorner<2,6>();
        jacobians[0]=jacobian_pose_i;
    }

    void update(Matrix3d Rs,double *Qs){
        Eigen::Quaterniond Qi(Qs[6], Qs[3], Qs[4], Qs[5]);
        Sophus::SO3d R0(Rs),R1(Qi);
        Vector3d delta_R=(R1.inverse()*R0).log();
        R=R*Sophus::SO3d::exp(delta_R).matrix();
    }
    MatrixXd Former(Matrix<double,6,1> parameters,Matrix<double,6,1> delta) const {
        Eigen::Vector3d Pi(parameters[0], parameters[1], parameters[2]);
        Sophus::SO3d Qi=Sophus::SO3d::exp(Vector3d(parameters[3], parameters[4],  parameters[5]));

        //distrub
        Pi=Pi+delta.head<3>();
        Qi=Qi*Sophus::SO3d::exp(delta.tail<3>());

        Sophus::SO3d Ri(Qi);
        Sophus::SO3d Rmeas(R);
        Vector3d nZ=-1.0*Vector3d::UnitZ();
        Vector2d res= (Rmeas * Ri.inverse()*nZ).head<2>();
        res=sqrt_info*res;
        //residual=sqrt_info*residual;
        return res;

    }

    bool check(double const *const *parameters,double **jacobians) const{
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Matrix<double,6,1> MatParm;
        MatParm<<Pi,Sophus::SO3d(Qi).log();

        double eps=1e-8;
        Matrix<double,6,1> delta;
        delta.setZero();
        Matrix<double,2,6> J;
        for (int i = 0; i <6 ; ++i) {
            delta(i)=eps;
            Matrix<double,2,1> delta_r=Former(MatParm,delta)-Former(MatParm,-delta);
            J.col(i)=delta_r/(2.0*eps);
            delta.setZero();
        }

        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);


        Matrix<double,2,6> Ja;
        Ja=jacobian_pose_i.topLeftCorner<2,6>();

        cout<<"zero test in RollPitchFactor"<<endl;
        cout<<J<<endl;
        cout<<"------------"<<endl;
        cout<<Ja<<endl;
        return true;
    }
    Matrix3d R;
    MatrixXd sqrt_info;
    int index;
    vector<MatrixXd> jacobians;
    VectorXd residual;
};
#endif //CS_VINS_ROLLPITCH_FACTOR_H
