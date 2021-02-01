
#ifndef VINS_ESTIMATOR_POSE_GRAPH_BUILDER_H
#define VINS_ESTIMATOR_POSE_GRAPH_BUILDER_H
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "pose_graph/keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph/pose_graph.h"
#include "parameters.h"
#include "pose_graph/keyframe.h"
#include <pangolin/pangolin.h>
#include "estimator.h"
#include "factor/pose_graph_factors.h"
#define SKIP_FIRST_CNT 10

using namespace std;

class PoseGraphBuilder{

public:
    queue<cvImgStampedConstPtr> image_buf;
    queue<Keyframe_pointsConstPtr> point_buf;
    queue<CombinedFactors*> pgfactor_buf;


public:
    PoseGraphBuilder(shared_ptr<Estimator> &estimator){
        last_t=Eigen::Vector3d::Zero();
        terminate=false;
        ofs_pose.open("./loop_pose_output.txt",fstream::out);
        if(!ofs_pose.is_open())
        {
            cerr << "ofs_pose is not open" << endl;
        }
        posegraph.loadVocabulary("/home/unicorn/Desktop/VINS-Course/config/brief_k10L6.bin");
        this->estimator=estimator;
        t_process=thread(&PoseGraphBuilder::process,this);
        t_process.detach();
        t_draw=thread(&PoseGraphBuilder::Draw,this);
        t_draw.detach();

    }
    ~PoseGraphBuilder(){
        ofs_pose.close();
//        pangolin::QuitAll();
    }
    void new_sequence();
    void GrabImg(cvImgStampedPtr &imgstamped);
    void GrabKeyframePoints(Keyframe_pointsPtr &kps);
    void GrabKeyFrameFactor(CombinedFactors *factor);
    void process();
    void Draw();
    std::mutex m_reloc_buf;
    std::queue<RelocinfoConstPtr> relo_buf;


public:
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    std::thread t_process,t_draw;
    std::mutex m_buf;
    std::mutex m_process;
    int frame_index  = 0;
    int sequence = 1;
    PoseGraph posegraph;
    KeyFrame *last_kf;
    int skip_first_cnt = 0;
    int SKIP_CNT;
    int skip_cnt = 0;
    bool load_flag = 0;
    bool start_flag = 0;
    double SKIP_DIS = 0;
    double last_image_time = -1;
    Eigen::Vector3d last_t;
    std::ofstream ofs_pose;
    shared_ptr<Estimator> estimator;
    bool terminate;
    std::mutex m_terminate;
    int last_index=0;
    CombinedFactors *accumFactor;
    CombinedFactors *currentFactor;
};
#endif //VINS_ESTIMATOR_POSE_GRAPH_BUILDER_H
