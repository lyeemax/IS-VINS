//
// Created by unicorn on 2020/12/28.
//

#ifndef CS_VINS_RELATIVE_POSE_FACTOR_H
#define CS_VINS_RELATIVE_POSE_FACTOR_H
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <utility/sophus_utils.hpp>
#include "pose_local_parameterization.h"
using namespace Eigen;
class RelativePoseFactor : public ceres::SizedCostFunction<6, 7, 7> {
public:

    RelativePoseFactor() = delete;

    RelativePoseFactor(const Vector3d delta_t_,const Matrix3d delta_R_) :delta_R(delta_R_),delta_t(delta_t_) {
        jacobians.resize(2);
        residual.resize(6,1);
    }
    void setIndex(int i, int j) {
        imu_i = i;
        imu_j = j;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        
        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);


        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        
        Matrix3d Rj(Qj),Ri(Qi);
        Vector3d res_t=delta_t-(Pj-Pi);
        Sophus::SO3d res_R(delta_R*Rj.transpose()*Ri);
        residual.head<3>()=res_t;
        residual.tail<3>()=res_R.log();

        residual=sqrt_info*residual;

        if (jacobians){
            Matrix3d J;
            Sophus::rightJacobianInvSO3(res_R.log(),J);
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3,3>(0,0)=Matrix3d::Identity();
                jacobian_pose_i.block<3,3>(3,3)=J;
                jacobian_pose_i=sqrt_info*jacobian_pose_i;
            }
            if(jacobians[1]){
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3,3>(0,0)=-Matrix3d::Identity();
                jacobian_pose_j.block<3,3>(3,3)=-J*Ri.transpose()*Rj;
                jacobian_pose_j=sqrt_info*jacobian_pose_j;

            }

        }

        //check(parameters,jacobians);
        return true;
    }

    void EvaluateOnlyJacobians(double const *PSi,double const *PSj,bool Nscheck=false){
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);

        Eigen::Vector3d Pj(PSj[0], PSj[1], PSj[2]);
        Eigen::Quaterniond Qj(PSj[6], PSj[3], PSj[4], PSj[5]);


        Matrix3d Rj(Qj),Ri(Qi);
        Vector3d res_t=delta_t-(Pj-Pi);
        Sophus::SO3d res_R(delta_R*Rj.transpose()*Ri);
        residual.head<3>()=res_t;
        residual.tail<3>()=res_R.log();

        Matrix3d J;
        Sophus::rightJacobianInvSO3(res_R.log(),J);
        Eigen::Matrix<double, 6, 6> jacobian_pose_i;
        jacobian_pose_i.setZero();
        jacobian_pose_i.block<3,3>(0,0)=Matrix3d::Identity();
        jacobian_pose_i.block<3,3>(3,3)=J;
        jacobians[0]=jacobian_pose_i;

        Eigen::Matrix<double, 6, 6> jacobian_pose_j;
        jacobian_pose_j.setZero();
        jacobian_pose_j.block<3,3>(0,0)=-Matrix3d::Identity();
        jacobian_pose_j.block<3,3>(3,3)=-J*Ri.transpose()*Rj;
        jacobians[1]=jacobian_pose_j;

    }

    void update(Vector3d ti,Matrix3d Ri,Vector3d tj,Matrix3d Rj,double *PSi,double *PSj){
//        Vector3d tij=Psj-Psi;
//        Matrix3d Rij=(Qi.inverse()*Qj).toRotationMatrix();
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);
        Eigen::Vector3d Pj(PSj[0], PSj[1], PSj[2]);
        Eigen::Quaterniond Qj(PSj[6], PSj[3], PSj[4], PSj[5]);
        Vector3d d_tj=Pj-tj;
        Vector3d d_ti=Pi-ti;
        Sophus::SO3d d_Rj=Sophus::SO3d(Qj.inverse()*Rj);
        Sophus::SO3d d_Ri=Sophus::SO3d(Qi.inverse()*Ri);

        delta_t+=d_tj-d_ti;
        MatrixXd Ji=-(Qj.inverse()*Qi).toRotationMatrix();
        delta_R=delta_R*Sophus::SO3d::exp(Ji*d_Ri.log()).matrix();
        delta_R=delta_R*Sophus::SO3d::exp(d_Rj.log()).matrix();
    }
    void shift(){
        imu_i--;
        imu_j--;
    }


    MatrixXd Former(Matrix<double,12,1> parameters,Matrix<double,12,1> delta) const {
        Eigen::Vector3d Pi(parameters[0], parameters[1], parameters[2]);
        Sophus::SO3d Qi=Sophus::SO3d::exp(Vector3d(parameters[3], parameters[4],  parameters[5]));
        Eigen::Vector3d Pj(parameters[6], parameters[7],parameters[8]);
        Sophus::SO3d Qj=Sophus::SO3d::exp(Vector3d(parameters[9], parameters[10], parameters[11]));

        //distrub
        Pi=Pi+delta.head<6>().head<3>();
        Qi=Qi*Sophus::SO3d::exp(delta.head<6>().tail<3>());
        Pj=Pj+delta.tail<6>().head<3>();
        Qj=Qj*Sophus::SO3d ::exp(delta.tail<6>().tail<3>());
        Eigen::Matrix<double, 6, 1> residual;
        Matrix3d Rj(Qj.matrix()),Ri(Qi.matrix());
        Vector3d res_t=delta_t-(Pj-Pi);
        Sophus::SO3d res_R(delta_R*Rj.transpose()*Ri);
        residual.head(3)=res_t;
        residual.tail(3)=res_R.log();
        residual=sqrt_info*residual;
        return residual;
    }

    bool check(double const *const *parameters,double **jacobians) const{
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Matrix<double,12,1> MatParm;
        MatParm<<Pi,Sophus::SO3d(Qi).log(),Pj,Sophus::SO3d(Qj).log();

        double eps=1e-8;
        Matrix<double,12,1> delta;
        delta.setZero();
        Matrix<double,6,12> J;
        for (int i = 0; i <12 ; ++i) {
            delta(i)=eps;
            Matrix<double,6,1> delta_r=Former(MatParm,delta)-Former(MatParm,-delta);
            J.col(i)=delta_r/(2.0*eps);
            delta.setZero();
        }

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

        Matrix<double,6,12> Ja;
        Ja.topLeftCorner<6,6>()=jacobian_pose_i.topLeftCorner<6,6>();
        Ja.bottomRightCorner<6,6>()=jacobian_pose_j.topLeftCorner<6,6>();

        cout<<"zero test in RelativePoseFactor"<<endl;
        cout<<Ja<<endl;
        cout<<"----------------------------"<<endl;
        cout<<J<<endl;
        return true;
    }



    Vector3d delta_t;
    Matrix3d delta_R;
    MatrixXd sqrt_info;
    vector<MatrixXd> jacobians;
    int imu_i,imu_j;
    VectorXd residual;
};

#endif //CS_VINS_RELATIVE_POSE_FACTOR_H
