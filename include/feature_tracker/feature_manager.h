#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
#include <iostream>
#include <mutex>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// #include <ros/console.h>
// #include <ros/assert.h>

#include "parameters.h"

class Feature
{
public:
  Feature(const Eigen::Matrix<double, 7, 1> &_point, double _td)
  {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    td=_td;
  }
  Vector3d point;
  Vector2d uv;
  Vector2d velocity;
  bool is_used;
  double td;
};

class IDFeatures
{
public:
  const int feature_id;
  int start_frame;
  vector<Feature> idfeatures; //idfeatures[0]是host
  bool tobeMarg;

  int used_num;
  bool is_outlier;//#TODO write this value
  double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
  bool significant;

  IDFeatures(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), is_outlier(false),estimated_depth(-1.0), solve_flag(0),significant(false),tobeMarg(false)
  {
  }

  int endFrame();
};

class FeatureManager
{
public:
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();
  bool goodFeature(IDFeatures &idfs);

  bool addFeatureAndCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  void debugShow();
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth(const VectorXd &x);
  VectorXd getDepthVector();
  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeNonkeyframe();
  void removeOutlier();
  std::mutex m_idfs;
  list<IDFeatures> IDsfeatures;
  int last_track_num;

private:
  double compensatedParallax2(const IDFeatures &idFeatures, int frame_count);
  const Matrix3d *Rs;
  Matrix3d ric[NUM_OF_CAM];
};

#endif