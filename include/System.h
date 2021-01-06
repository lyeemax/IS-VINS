#pragma once

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

#include <fstream>
#include <condition_variable>
#include <pangolin/pangolin.h>

#include "estimator.h"
#include "parameters.h"
#include "feature_tracker/feature_tracker_simple.h"
#include "pose_graph/pose_graph_builder.h"

class System
{
public:
    System(std::string sConfig_files);

    ~System();

    void PubImageData(double dStampSec, cv::Mat &img);

    void PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
        const Eigen::Vector3d &vAcc);

    bool initialized(){
        return estimator->solver_flag==Estimator::SolverFlag::NON_LINEAR;
    }


    // thread: visual-inertial odometry
    void ProcessBackEnd();
    void Draw();
    
    pangolin::OpenGlRenderState s_camp;
    pangolin::View d_camp;

    FeatureTracker trackerData[NUM_OF_CAM];

private:

    double first_image_time;
    int pub_count = 1;
    bool first_image_flag = true;
    double last_image_time = 0;
    bool init_pub = 0;

    //estimator
    shared_ptr<Estimator> estimator;
    shared_ptr<PoseGraphBuilder> pgbuilder;

    std::condition_variable con;
    double current_time = -1;
    std::queue<ImuConstPtr> imu_buf;
    std::queue<ImgConstPtr> feature_buf;
    int sum_of_wait = 0;

    std::mutex m_buf;

    std::mutex m_estimator;


    bool init_feature = 0;
    double last_imu_t = 0;
    std::vector<Eigen::Vector3d> vPath_to_draw;
    bool bStart_backend;
    std::ofstream ofs_pose;
    std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
    
};
