# IS-VINS
## Information Sparsification for Visual Inertial Navigation System (IS-VINS)

**IS-VINS** is a sparisification-based system which extends [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) to consistent optimization of **VO & VIO & Pose-Graph**; This system is motivated by preserving sparse and nonlinear information after marginalization and simplifying VIO scheme rather than preserving linearied prior that ordinary fixed-lag smoother shares;Thanks to sparisification,pose graph is builted upon information structure to achieve consistent optimazation;This code supports ***Linux without ROS***, it is possible to run on **Mac OS X** or **Windows** with sightly change.Please cite us if you use our code.


Followings are main contributions in this work:

1. Frontend is optic flow based method just like VINS-MONO but simplified.

2. VIO is performed in two stages marginalization and sparsification,which turns VIO to be a combanation of VO and VIO.

3. Pose graph optimization reuses information of VO with gravity observation and relative pose prior and performs with loop-closure information. So it's able to evaluate covariance of camera pose online. 

<img src="https://github.com/lyeemax/IS-VINS/blob/GL_IS_VINS/others/%E4%BF%A1%E6%81%AF%E7%9F%A9%E9%98%B5.png" alt="sparsification" width="480" height="360" border="10" /><img src="https://github.com/lyeemax/IS-VINS/blob/GL_IS_VINS/others/Screenshot%20from%202021-03-26%2016-56-43.png" alt="euroc_MH_05" width="480" height="360" border="10" />


***Authors：***: Jixiang Ma(unicorn@hust.edu.cn)

Installation
------------
We tested this code with ubuntu 18.04 with Ceres 2.0.0,Eigen 3.3.4,OpenCV 3.2.0 and Pangolin.
```
  $ git clone https://github.com/lyeemax/IS-VINS.git
  $ mkdir build  
  $ cd build/  
  $ cmake ..
  $ make -j 9  
```

Evaluate Euroc Dataset
------------
Download EuRoc Mav dataset and extract zips. Open a terminal with parameters of dataset path and config files:
```
  $ ./run_euroc PATH_TO_EUROC/mav0/ ../config/ 
```

