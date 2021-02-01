#ifndef CS_VINS_POSE_GRAPH_FACTORS_H
#define CS_VINS_POSE_GRAPH_FACTORS_H
#include "factor/relative_pose_factor.h"
#include "factor/rollpitch_factor.h"

struct CombinedFactors{
    RelativePoseFactor *relativePoseFactor;
    RollPitchFactor *rollPitchFactor;
    long vio_index;
    int length;
    long pg_index;
    Eigen::MatrixXd covRel;
    Eigen::MatrixXd covAbs;
    double distance;
    double ts;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    CombinedFactors(long index=0){
        vio_index=-1;
        pg_index=index;
        length=0;
        covRel=Eigen::Matrix<double,6,6>::Zero();
        relativePoseFactor=new RelativePoseFactor(Vector3d(0,0,0),Eigen::Matrix3d::Identity());
    }

    CombinedFactors operator+(CombinedFactors other){
        Sophus::SE3d T0(relativePoseFactor->delta_R,relativePoseFactor->delta_t);
        Sophus::SE3d T1(other.relativePoseFactor->delta_R,other.relativePoseFactor->delta_t);
        Eigen::MatrixXd covRel1=  (other.relativePoseFactor->sqrt_info.transpose()*other.relativePoseFactor->sqrt_info).inverse();
        covRel=covRel+T0.Adj()*covRel1*T0.Adj().transpose();
        rollPitchFactor=other.rollPitchFactor;




        delete relativePoseFactor;
        relativePoseFactor=new RelativePoseFactor((T0*T1).translation(),(T0*T1).rotationMatrix());
        relativePoseFactor->sqrt_info=Eigen::LLT<MatrixXd>(covRel.inverse()).matrixL().transpose();

        distance=(T0*T1).translation().norm();
        length++;

        if(vio_index==-1){
            t=other.t;
            R=other.R;
            vio_index=other.vio_index;
            ts=other.ts;
        }
        return *this;
    }
};
#endif //CS_VINS_POSE_GRAPH_FACTORS_H
