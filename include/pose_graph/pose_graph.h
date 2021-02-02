#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <stdio.h>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "DBoW/DBoW2.h"
#include "DVision/DVision.h"
#include "DBoW/TemplatedDatabase.h"
#include "DBoW/TemplatedVocabulary.h"
#include "factor/pose_graph_factors.h"
#include "factor/pose_local_parameterization.h"
#include "factor/relative_pose_factor.h"

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();
	~PoseGraph();
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
	void loadVocabulary(std::string voc_path);
	void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1 > &_loop_info);
	KeyFrame* getKeyFrame(int index);
	Vector3d t_drift;
	double yaw_drift;
	Matrix3d r_drift;
	// world frame( base sequence or first sequence)<----> cur sequence frame  
	Vector3d w_t_vio;
	Matrix3d w_r_vio;


public:
	int detectLoop(KeyFrame* keyframe, int frame_index);
	void addKeyFrameIntoVoc(KeyFrame* keyframe);
	void optimizeCS();
	list<KeyFrame*> keyframelist;
	std::mutex m_keyframelist;
	std::mutex m_optimize_buf;
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;
	std::queue<int> optimize_buf;

	int global_index;
	int sequence_cnt;
	vector<bool> sequence_loop;
	map<int, cv::Mat> image_pool;
	int earliest_loop_index;
	int base_sequence;

	BriefDatabase db;
	BriefVocabulary* voc;
};