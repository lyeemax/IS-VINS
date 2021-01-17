#pragma once
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"
#include <vector>
using namespace Eigen;
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void EvaluateOnlyJacobians(double const *PSi,double const *PSj,double const *PSic,double const *inv_dep,bool Nscheck=false);
    double EvaluateResidual(double const *PSi,double const *PSj,double const *PSic,double const *inv_dep){
        Eigen::Vector3d Pi(PSi[0], PSi[1], PSi[2]);
        Eigen::Quaterniond Qi(PSi[6], PSi[3], PSi[4], PSi[5]);

        Eigen::Vector3d Pj(PSj[0], PSj[1], PSj[2]);
        Eigen::Quaterniond Qj(PSj[6], PSj[3], PSj[4], PSj[5]);

        Eigen::Vector3d tic(PSic[0], PSic[1], PSic[2]);
        Eigen::Quaterniond qic(PSic[6], PSic[3], PSic[4], PSic[5]);

        double inv_dep_i =inv_dep[0];

        Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        Eigen::Vector2d residual;

        double dep_j = pts_camera_j.z();
        residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();

        double res=residual.transpose()*residual;
        return res;
    }
    void check(double **parameters);
    void setIndex(int i,int j,int f){
        imu_i=i;
        imu_j=j;
        feature_idx=f;
    }
    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
    int imu_i,imu_j,feature_idx;
    std::vector<Eigen::MatrixXd> jacobians;
    bool outlier;
    VectorXd residual;

};
