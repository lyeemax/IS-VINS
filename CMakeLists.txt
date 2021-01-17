cmake_minimum_required(VERSION 2.8.3)
project(IS_Vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_REQUIRED_FLAGS "-Wno-reorder")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
find_package(Sophus)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/thirdparty
  ${PROJECT_SOURCE_DIR}/thirdparty/camera_models/include
		${Sophus_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(camera_model SHARED
		thirdparty/camera_models/src/camera_models/chessboard/Chessboard.cc
		thirdparty/camera_models/src/camera_models/calib/CameraCalibration.cc
		thirdparty/camera_models/src/camera_models/camera_models/Camera.cc
		thirdparty/camera_models/src/camera_models/camera_models/CameraFactory.cc
		thirdparty/camera_models/src/camera_models/camera_models/CostFunctionFactory.cc
		thirdparty/camera_models/src/camera_models/camera_models/PinholeCamera.cc
		thirdparty/camera_models/src/camera_models/camera_models/CataCamera.cc
		thirdparty/camera_models/src/camera_models/camera_models/EquidistantCamera.cc
		thirdparty/camera_models/src/camera_models/camera_models/ScaramuzzaCamera.cc
		thirdparty/camera_models/src/camera_models/sparse_graph/Transform.cc
		thirdparty/camera_models/src/camera_models/gpl/gpl.cc
		thirdparty/camera_models/src/camera_models/gpl/EigenQuaternionParameterization.cc)

add_library(reloc SHARED
		thirdparty/DBoW/BowVector.cpp
		thirdparty/DBoW/FBrief.cpp
		thirdparty/DBoW/FeatureVector.cpp
		thirdparty/DBoW/QueryResults.cpp
		thirdparty/DBoW/ScoringObject.cpp
		thirdparty/DUtils/Random.cpp
		thirdparty/DUtils/Timestamp.cpp
		thirdparty/DVision/BRIEF.cpp
		thirdparty/VocabularyBinary.cpp)

add_library(ceres_factor SHARED
		include/factor/imu_factor.h
		include/factor/integration_base.h
		include/factor/pose_local_parameterization.h
		include/factor/projection_factor.h
		src/factor/pose_local_parameterization.cpp
		src/factor/projection_factor.cpp
		include/factor/relative_pose_factor.h
		include/factor/se3_prior_factor.h
		include/factor/linear9_factor.h
		include/factor/rollpitch_factor.h
		include/factor/yaw_factor.h
		include/factor/pose_graph_factors.h
		#include/factor/imu_bias_factor.h
		)

target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

target_link_libraries(reloc camera_model)

target_link_libraries(ceres_factor ${CERES_LIBRARIES})

ADD_LIBRARY(ISVins SHARED
    src/System.cpp
    src/parameters.cpp
    src/estimator.cpp
	src/feature_tracker/feature_manager.cpp
    src/utility/utility.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
	src/feature_tracker/feature_tracker_simple.cpp
	src/pose_graph/keyframe.cpp
	src/pose_graph/pose_graph.cpp
	src/pose_graph/pose_graph_builder.cpp
	include/pose_graph/pose_graph_builder.h include/msg/msgtype.h
		#include/utility/factor_descent.h
		include/utility/eigen_file.h)

target_link_libraries(ISVins
  ${OpenCV_LIBS}  
  ${CERES_LIBRARIES}
  ${Pangolin_LIBRARIES}
  camera_model
		ceres_factor
		reloc)

add_executable(run_euroc test/run_euroc.cpp )
target_link_libraries(run_euroc
  ISVins -lpthread)
