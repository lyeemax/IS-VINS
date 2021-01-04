# IS-VINS
## Information Sparsification for Visual Inertial Navigation System (IS-VINS)

**IS-VINS** is a sparisification-based system which extends [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) to consistent optimization of **VO & VIO & Pose-Graph**; This system is motivated by preserving sparse and nonlinear information after marginalization and simplifying VIO scheme rather than preserving linearied prior that ordinary fixed-lag smoother shares;Thanks to sparisification,pose graph is builted upon information structure to achieve consistent optimazation;This code supports ***Linux without ROS***, it is possible to run on **Mac OS X** or **Windows** with sightly change.Please cite us if you use our code.

***To repay the SLAM community, please feel free to ask me for the detail before the paper released.***

--:Followings are some important work in this project:

1. Frontend is optic flow based method just like VINS-MONO but simplified.

2. VIO stage is performed in two stage marginalization and sparsification,which turns VIO to be a combanation of VO with gravity observation and VIO.

3. Pose graph optimization reuses information of VO with gravity observation and performs with loop-closure information. So it's able to fetch covariance of viechle or camera online. 

***Authors***: Jixiang Ma(unicorn@hust.edu.cn)

**Related Papers**
**To be released soon**, Jixiang Ma, Suijun Zheng,Yong Xie. IEEE/RSJ Robotics and Automation Letters,In Press.


