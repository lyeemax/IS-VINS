
#ifndef CS_VINS_SE3_PRIOR_FACTOR_H
#define CS_VINS_SE3_PRIOR_FACTOR_H
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <utility/sophus_utils.hpp>
using namespace Eigen;
class SE3PriorFactor : public ceres::SizedCostFunction<6, 7> {
public:
    SE3PriorFactor() = delete;

    SE3PriorFactor(const Vector3d t_new, const Quaterniond R_new) : R(R_new), t(t_new) {
        linearized=false;
        jacobians.resize(1);
        residual.resize(6,1);
    }
    void setIndex(int i) {
        index=i;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        
        Sophus::SO3d ri(Qi);
        Sophus::SO3d rp(R);
        Sophus::SO3d res_r = rp.inverse() * ri;
        residual.block<3,1>(3,0) = res_r.log();
        residual.block<3,1>(0,0) = Pi - t;
        residual=sqrt_info*residual;
//        double error=residual.transpose()*residual;
//        if(error>1){
//            cerr<<"residual of absolute pose of "<<index<<" +++: "<<error<<endl;
//        }

        if (jacobians){
            if (jacobians[0]) {
                if(linearized){
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i.topLeftCorner<6,6>()=this->jacobians[0];
                    jacobian_pose_i=sqrt_info*jacobian_pose_i;
                }else{
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i.block<3,3>(0,0) = Matrix3d::Identity();
                    Matrix3d J;
                    Sophus::rightJacobianInvSO3(res_r.log(),J);
                    jacobian_pose_i.block<3,3>(3,3) =J;

                    jacobian_pose_i=sqrt_info*jacobian_pose_i;
                }

            }

        }

       //check(parameters,jacobians);
        return true;
    }

    void EvaluateOnlyJacobians(double const *PSi,bool Nscheck=false){
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);

        Sophus::SO3d ri(Qi);
        Sophus::SO3d rp(R);
        Sophus::SO3d res_r = rp.inverse() * ri;
        residual.block<3,1>(3,0) = res_r.log();
        residual.block<3,1>(0,0) = Pi - t;
        Eigen::Matrix<double, 6, 6> jacobian_pose_i;
        jacobian_pose_i.setZero();
        jacobian_pose_i.block<3,3>(0,0) = Matrix3d::Identity();
        Matrix3d J;
        Sophus::rightJacobianInvSO3(res_r.log(),J);
        jacobian_pose_i.block<3,3>(3,3) = J;
        jacobians[0]=jacobian_pose_i;
        if(!Nscheck){
            linearized=0;
        }
    }

    MatrixXd Former(Matrix<double,6,1> parameters,Matrix<double,6,1> delta) const {
        Eigen::Vector3d Pi(parameters[0], parameters[1], parameters[2]);
        Sophus::SO3d Qi=Sophus::SO3d::exp(Vector3d(parameters[3], parameters[4],  parameters[5]));

        //distrub
        Pi=Pi+delta.head<3>();
        Qi=Qi*Sophus::SO3d::exp(delta.tail<3>());


        Eigen::Matrix<double, 6, 1> residual;

        Sophus::SO3d ri(Qi);
        Sophus::SO3d rp(R);
        Sophus::SO3d res_r = rp.inverse() * ri;
        residual.block<3,1>(3,0) = res_r.log();
        residual.block<3,1>(0,0) = Pi - t;
        //residual=sqrt_info*residual;
        return residual;

    }

    bool check(double const *const *parameters,double **jacobians) const{
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Matrix<double,6,1> MatParm;
        MatParm<<Pi,Sophus::SO3d(Qi).log();

        double eps=1e-8;
        Matrix<double,6,1> delta;
        delta.setZero();
        Matrix<double,6,6> J;
        for (int i = 0; i <6 ; ++i) {
            delta(i)=eps;
            Matrix<double,6,1> delta_r=Former(MatParm,delta)-Former(MatParm,-delta);
            J.col(i)=delta_r/(2.0*eps);
            delta.setZero();
        }

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);


        Matrix<double,6,6> Ja;
        Ja=jacobian_pose_i.topLeftCorner<6,6>();


        auto zero=Ja-J;

        cout<<"zero test in SE3PriorFactor"<<endl;
        cout<<J<<endl;
        cout<<"------------"<<endl;
        cout<<Ja<<endl;
        return true;
    }

    Vector3d t;
    Matrix3d R;
    MatrixXd sqrt_info;
    int index;
    vector<MatrixXd> jacobians;
    bool linearized;
    VectorXd residual;
};
#endif //CS_VINS_SE3_PRIOR_FACTOR_H
