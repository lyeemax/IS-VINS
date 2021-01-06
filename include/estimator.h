#pragma once

#include "parameters.h"
#include "feature_tracker/feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "factor/integration_base.h"
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include "pose_graph/pose_graph.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/relative_pose_factor.h"
#include "factor/se3_prior_factor.h"
#include "factor/linear9_factor.h"
#include "factor/imu_factor.h"
#include <msg/msgtype.h>
#include "factor//rollpitch_factor.h"
#include "factor/yaw_factor.h"
#include "factor/pose_graph_factors.h"

static long PoseGraphFactorCount=0;
class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header);

    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);
    // internal
    void clearState();
    bool initialStructure();
    bool checkIMUExcitation();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void backendOptimization();

    void problemSolve();
    void initFactorGraph();
    void MargForward();
    void MargBackward();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR,
        INITIAL_STRUCTURE
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_NEW = 1
    };
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[ALL_BUF_SIZE];
    Vector3d Vs[(ALL_BUF_SIZE)];
    Matrix3d Rs[(ALL_BUF_SIZE)];
    Vector3d Bas[(ALL_BUF_SIZE)];
    Vector3d Bgs[(ALL_BUF_SIZE)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(ALL_BUF_SIZE)];

    IntegrationBase *pre_integrations[(ALL_BUF_SIZE)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(ALL_BUF_SIZE)];
    vector<Vector3d> linear_acceleration_buf[(ALL_BUF_SIZE)];
    vector<Vector3d> angular_velocity_buf[(ALL_BUF_SIZE)];

    int frame_count;
    int sum_of_back, sum_of_front;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool failure_occur;

    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[ALL_BUF_SIZE][SIZE_POSE];
    double para_SpeedBias[ALL_BUF_SIZE][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Td[1][1];

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
    std::mutex m_loop_buf;
    std::queue<pair<int,Eigen::Matrix<double,8,1>>> loop_buf;
    std::mutex m_pose_graph_buf;
    std::queue<CombinedFactors*> pose_graph_factors_buf;

    //system factor
    Linear9Factor* vioVBPrior;
    vector<RelativePoseFactor*> vioRelativePoseEdges;
    vector<RollPitchFactor*> vioRollPitchEdges;
    SE3PriorFactor* vioPosePriorEdge;


    //forward prior
    vector<int> MargPointIdx;
    vector<IDFeatures> features2Marg;
    vector<ProjectionFactor*> forwardProjectiontoSparsify;

    //forward result
    SE3PriorFactor* forwardPosePriorEdgeToAdd;

    //backward prior
    IMUFactor* backwardIMUtoSparsify;

    //backward result
    Linear9Factor* backwardVBEdgeToAdd;
    RelativePoseFactor* backwardRelativePoseEdgeToAdd;

public:
    std::mutex m_front_terminated;
    bool front_terminated;
};
