
#ifndef VINS_ESTIMATOR_MSGTYPE_H
#define VINS_ESTIMATOR_MSGTYPE_H

#include<Eigen/Core>
#include <vector>
#include <memory>

//imu for vio
struct IMU_MSG
{
    double header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};
typedef std::shared_ptr<IMU_MSG const> ImuConstPtr;

//image for vio
struct IMG_MSG {
    double header;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> id_of_point;
    std::vector<float> u_of_point;
    std::vector<float> v_of_point;
    std::vector<float> velocity_x_of_point;
    std::vector<float> velocity_y_of_point;
};
typedef std::shared_ptr <IMG_MSG const > ImgConstPtr;

struct relocation_infos{
    double header;
    std::vector<Eigen::Vector2d> uv_old_norm;
    std::vector<double> matched_id;
    Eigen::Vector3d t_old;
    Eigen::Matrix3d R_old;
    int index;
};
typedef  std::shared_ptr<const relocation_infos> RelocinfoConstPtr;
typedef  std::shared_ptr<relocation_infos> RelocinfoPtr;

struct keyframe_points{
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector2d> uv_points;
    std::vector<Eigen::Vector2d> norm_points;
    std::vector<int> vids;
    double header;
};
typedef std::shared_ptr<keyframe_points> Keyframe_pointsPtr;
typedef std::shared_ptr<const keyframe_points> Keyframe_pointsConstPtr;

struct PoseStamped{
    double header;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    PoseStamped(double _header,Eigen::Matrix3d _R,Eigen::Vector3d _t):
    header(_header),R(_R),t(_t){};
};
typedef std::shared_ptr<PoseStamped> PosestampedPtr;
typedef std::shared_ptr<const PoseStamped> PosestampedConstPtr;

struct cvImgStamped{
    double header;
    cv::Mat img;
    cvImgStamped(double _header,cv::Mat _img):header(_header),img(_img){};
};
typedef std::shared_ptr<cvImgStamped> cvImgStampedPtr;
typedef std::shared_ptr<const cvImgStamped> cvImgStampedConstPtr;
#endif //VINS_ESTIMATOR_MSGTYPE_H
