#include "estimator.h"
#include <ostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "utility/eigen_file.h"
using namespace Eigen;
double eps=1e-16;

Estimator::Estimator() : f_manager{Rs}
{
    for (size_t i = 0; i < ALL_BUF_SIZE; i++)
    {
        pre_integrations[i] = nullptr;
    }
    for(auto &it: all_image_frame)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration = nullptr;
    vioRelativePoseEdges.resize(Vo_SIZE);//window size relative position, vioRelativePoseEdges[0] is empty
    
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = PIXEL_SQRT_INFO* Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < ALL_BUF_SIZE; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = true,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;
    front_terminated= false;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    
    tmp_pre_integration = nullptr;

    f_manager.clearState();
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (first_imu)
    {
        first_imu = false;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    if (f_manager.addFeatureAndCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_NEW;
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if (ESTIMATE_EXTRINSIC == 2)
    {
        cout << "calibrating extrinsic param, rotation movement is needed" << endl;
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == ALL_BUF_SIZE-1)//fill all already
        {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
            {
                result = initialStructure();
                initial_timestamp = header;
            }
            if (result)
            {
                solver_flag = INITIAL_STRUCTURE;
                cout << "Initialization finish!" << endl;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();

                last_R = Rs[ALL_BUF_SIZE-1];
                last_P = Ps[ALL_BUF_SIZE-1];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else if(solver_flag==NON_LINEAR)
    {
        TicToc t_solve;
        solveOdometry();

        if (failureDetection())
        {
            failure_occur = 1;
            clearState();
            setParameter();
            cout<<"detected failure"<<endl;
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= ALL_BUF_SIZE-1; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[ALL_BUF_SIZE-1];
        last_P = Ps[ALL_BUF_SIZE-1];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

bool Estimator::checkIMUExcitation() {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
        double dt = frame_it->second.pre_integration->sum_dt;
        Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
        var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));
    if (var < 0.25)
    {
        cerr<<"IMU excitation not enouth!"<<endl;
        return false;
    }else
        return true;
}
bool Estimator::initialStructure()
{
    TicToc t_sfm;
   if(!checkIMUExcitation())
       return false;
    // global sfm
    Quaterniond Q[ALL_BUF_SIZE];
    Vector3d T[ALL_BUF_SIZE];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &idFeatures : f_manager.IDsfeatures)
    {
        int imu_j = idFeatures.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = idFeatures.feature_id;
        for (auto &feature : idFeatures.idfeatures)
        {
            imu_j++;
            Vector3d pts_j = feature.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int fineframe;
    if (!relativePose(relative_R, relative_T, fineframe))
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, fineframe,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        cout << "global SFM failed!" << endl;
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= ALL_BUF_SIZE-1; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &features : f_manager.IDsfeatures)
    {
        features.used_num = features.idfeatures.size();
        if (!f_manager.goodFeature(features))
            continue;
        features.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < ALL_BUF_SIZE-2; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, ALL_BUF_SIZE-1);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
//                printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < ALL_BUF_SIZE-1)
        return;
    if (solver_flag != INITIAL)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);//why only use position?? featureMannger share Rs with Esitimaster
        //cout << "triangulation costs : " << t_tri.toc() << endl;        
        backendOptimization();
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= ALL_BUF_SIZE-1; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();

    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));//Rba //before -> after
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        //ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    vioVBPrior->VB.tail<3>()=rot_diff*vioVBPrior->VB.tail<3>();
    vioPosePriorEdge->R=rot_diff*vioPosePriorEdge->R;

    for (int i = 0; i <= ALL_BUF_SIZE-1; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    bool failed=false;
    if (f_manager.last_track_num < 2)
    {
        //ROS_INFO(" little IDsfeatures %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[ALL_BUF_SIZE-1].norm() > 2.5)
    {
        cout<<" big IMU acc bias estimation %f"<<endl<< Bas[ALL_BUF_SIZE-1].norm()<<endl;
        failed= true;
    }
    if (Bgs[ALL_BUF_SIZE-1].norm() > 1.0)
    {
        cout<<" big IMU gyr bias estimation %f"<<endl<< Bgs[ALL_BUF_SIZE-1].norm()<<endl;
        failed= true;
    }
    /*
    if (tic(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[ALL_BUF_SIZE-1];
    if ((tmp_P - last_P).norm() > 5)
    {
        cout<<" big translation"<<endl;
        //failed= true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        cout<<" big z translation"<<endl;
        //failed= true;
    }
    Matrix3d tmp_R = Rs[ALL_BUF_SIZE-1];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        cout<<" big delta_angle "<<endl;
        //failed= true;
    }
    if(failed){
        //debug
        cout<<"debug roll pitch factor"<<endl;
        for (int i = 0; i <vioRollPitchEdges.size() ; ++i) {
            auto factor=vioRollPitchEdges[i];
            cout<<"roll pitch of index :"<<factor->index<<" with info "<<endl;
            cout<<factor->sqrt_info.transpose()*factor->sqrt_info<<endl;
        }

        cout<<"debug relative pose factor"<<endl;
        for (int i = 1; i <vioRelativePoseEdges.size() ; ++i) {
            auto factor=vioRelativePoseEdges[i];
            cout<<"relative pose of index :"<<factor->imu_i<<" and "<<factor->imu_j<<" with info "<<endl;
            cout<<factor->sqrt_info.transpose()*factor->sqrt_info<<endl;
        }
        cout<<"debug  pose prior factor"<<endl;
        cout<<vioPosePriorEdge->sqrt_info.transpose()*vioPosePriorEdge->sqrt_info<<endl;

        cout<<"debug speed and bias prior factor"<<endl;
        cout<<vioVBPrior->sqrt_info.transpose()*vioVBPrior->sqrt_info<<endl;
        //return true;
    }
    return false;
}

void Estimator::initFactorGraph(){

    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership=ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problemOptions);
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    vector<IMUFactor*> factors2Sparsify;

    for (int i = 0; i < ALL_BUF_SIZE; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

    for (int i = 0; i < ALL_BUF_SIZE-1; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor=new IMUFactor(pre_integrations[j]);
        if(i < Vo_SIZE - 1){
            imu_factor->setIndex(i,j);
            factors2Sparsify.push_back(imu_factor);
        }
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    int feature_index = -1;
    for (auto &idFeatures : f_manager.IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!f_manager.goodFeature(idFeatures))
            continue;

        ++feature_index;

        int imu_i = idFeatures.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = idFeatures.idfeatures[0].point;
        for (auto &feature : idFeatures.idfeatures)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = feature.point;

            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            f->setIndex(imu_i,imu_j,feature_index);
            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS*3;
    options.max_solver_time_in_seconds = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    //OrderMap
    //T0 T1.....Tvo VBvo VB0 VB1... VBv0-1
    std::unordered_map<double *,pair<int,int>> OrderMap;
    int idx=0;
    for (int i = 0; i < Vo_SIZE ; ++i) {
        OrderMap[para_Pose[i]]=make_pair(idx,6);
        idx+=6;
    }
    OrderMap[para_SpeedBias[Vo_SIZE - 1]]=make_pair(idx, 9);
    idx+=9;
    for (int i = 0; i < Vo_SIZE - 1 ; ++i) {
        OrderMap[para_SpeedBias[i]]=make_pair(idx,9);
        idx+=9;
    }

//    std::unordered_map<double *,pair<int,int>> OrderMap;
//    int idx=0;
//    for (int i = 0; i < Vo_SIZE ; ++i) {
//        OrderMap[para_Pose[i]]=make_pair(idx,6);
//        idx+=6;
//        OrderMap[para_SpeedBias[i]]=make_pair(idx,9);
//        idx+=9;
//    }


    MatrixXd Lamda;
    Lamda.resize(Vo_SIZE * 15, Vo_SIZE * 15);
    Lamda.setZero();
    assert(factors2Sparsify.size() == Vo_SIZE - 1);
    for (int i = 0; i <factors2Sparsify.size() ; ++i) {
        auto imufactor= factors2Sparsify[i];
        int imu_i=imufactor->imu_i;
        int imu_j=imufactor->imu_j;
        imufactor->Evaluate(para_Pose[imu_i],para_SpeedBias[imu_i],para_Pose[imu_j],para_SpeedBias[imu_j]);
        vector<MatrixXd> jacobiansI=imufactor->jacobians;
        MatrixXd omegaI=imufactor->sqrt_info.transpose()*imufactor->sqrt_info;
        //cout<<"my omegaI"<<endl<<omegaI.eigenvalues().real()<<endl;
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_SpeedBias[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        ParamMap.push_back(para_SpeedBias[imu_j]);
        for (int j = 0; j <ParamMap.size() ; ++j) {
            auto v_j=OrderMap[ParamMap[j]];
            int index_j=v_j.first;
            int dim_j=v_j.second;
            MatrixXd JtW = jacobiansI[j].transpose() * omegaI;
            for (int k = j; k <ParamMap.size() ; ++k) {
                auto v_k=OrderMap[ParamMap[k]];
                int index_k=v_k.first;
                int dim_k=v_k.second;
                MatrixXd hessian = JtW * jacobiansI[k];
                Lamda.block(index_j, index_k, dim_j, dim_k).noalias() += hessian;
                if (j != k) {
                    Lamda.block(index_k, index_j, dim_k, dim_j).noalias() += hessian.transpose();
                }
            }
        }
    }


    //cout<<Lamda.eigenvalues().real()<<endl;

    int t_psvb_dim= Vo_SIZE * 6 + 9;
    int t_vbs_dim= (Vo_SIZE - 1) * 9;

    Eigen::MatrixXd Lamda_rr=Lamda.block(0, 0, t_psvb_dim, t_psvb_dim);
    Eigen::MatrixXd Lamda_mm=Lamda.block(t_psvb_dim, t_psvb_dim, t_vbs_dim, t_vbs_dim);

    Eigen::MatrixXd Lamda_mm_inv = Lamda_mm.fullPivLu().solve(Eigen::MatrixXd::Identity(t_vbs_dim, t_vbs_dim));
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, t_psvb_dim, t_psvb_dim, t_vbs_dim);
    Eigen::MatrixXd Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();
    assert(Lamda_prior.rows() == Vo_SIZE * 6 + 9);

    int asize=Vo_SIZE * 6 + 9;
    //keep relative position factor
    vioRelativePoseEdges[0]= nullptr;
    for (int i = 0; i < Vo_SIZE - 1 ; ++i) {
        int j = i + 1;

        Vector3d Psj,Psi;
        Psi<<para_Pose[i][0], para_Pose[i][1], para_Pose[i][2];
        Psj<<para_Pose[j][0], para_Pose[j][1], para_Pose[j][2];
        Quaterniond Qi(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
        Quaterniond Qj(para_Pose[j][6], para_Pose[j][3], para_Pose[j][4], para_Pose[j][5]);
        Vector3d tij=Psj-Psi;
        Matrix3d Rij=(Qi.inverse()*Qj).toRotationMatrix();

        RelativePoseFactor* relativePoseFactor(new RelativePoseFactor(tij,Rij));
        relativePoseFactor->setIndex(i,j);
        relativePoseFactor->EvaluateOnlyJacobians(para_Pose[i],para_Pose[j]);
        vioRelativePoseEdges[j]=relativePoseFactor;
    }

    //keep absolute pose for the first pose
    Vector3d Ps0;
    Ps0<<para_Pose[0][0], para_Pose[0][1], para_Pose[0][2];
    Quaterniond Q0(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]);
    SE3PriorFactor* se3PriorFactor(new SE3PriorFactor(Ps0,Q0));

    se3PriorFactor->EvaluateOnlyJacobians(para_Pose[0]);
    se3PriorFactor->setIndex(0);
    vioPosePriorEdge=se3PriorFactor;

    //keep last velocity and bias factor
    int last= Vo_SIZE - 1;
    Matrix<double,9,1> VB;
    VB << para_SpeedBias[last][0], para_SpeedBias[last][1], para_SpeedBias[last][2],
            para_SpeedBias[last][3], para_SpeedBias[last][4], para_SpeedBias[last][5],
            para_SpeedBias[last][6], para_SpeedBias[last][7], para_SpeedBias[last][8];
    Linear9Factor* VBPriorfactor(new Linear9Factor(VB));
    VBPriorfactor->setIndex(last);
    VBPriorfactor->EvaluateOnlyJacobians(para_SpeedBias[last]);
    vioVBPrior=VBPriorfactor;


    //remove VB0---VB_last-1
    for (int i = 0; i < Vo_SIZE - 1 ; ++i) {
        for(auto iter=OrderMap.begin();iter!=OrderMap.end();){
            if(iter->first==para_SpeedBias[i]){
                iter=OrderMap.erase(iter);
                break;
            }else{
                iter++;
            }
        }
    }

    //restore information matrix for each factor
    Eigen::MatrixXd Jr;
    Jr.resize(6 * Vo_SIZE + 9, 6 * Vo_SIZE + 9);
    Jr.setZero();
    int rows=0;

    for (int i = 1; i < vioRelativePoseEdges.size() ; ++i) {
        auto factor=vioRelativePoseEdges[i];
        int Jrow=factor->num_residuals();
        auto jacobians=factor->jacobians;
        int imu_i=factor->imu_i;
        int imu_j=factor->imu_j;
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);

        for (int j=0;j<jacobians.size();j++) {
            auto v_j=OrderMap[ParamMap[j]];
            int idx=v_j.first;
            int dim = v_j.second;
            Jr.block(rows,idx,Jrow,dim)+=jacobians[j];
        }
        rows+=Jrow;
    }

    {
        int Jrow=vioPosePriorEdge->num_residuals();;
        auto v_j=OrderMap[para_Pose[vioPosePriorEdge->index]];
        int idx=v_j.first;
        int dim = v_j.second;
        Jr.block(rows,idx,Jrow,dim)+=vioPosePriorEdge->jacobians[0];
        rows+=Jrow;
    }
    {
        int Jrow=vioVBPrior->num_residuals();;
        auto v_j=OrderMap[para_SpeedBias[vioVBPrior->index]];
        int idx=v_j.first;
        int dim = v_j.second;
        Jr.block(rows,idx,Jrow,dim)+=vioVBPrior->jacobians[0];
        rows+=Jrow;
    }

//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
//    if(Jqr.rank()<Jr.cols()){
//        cerr<<"Jacobian of initial recovered factor is rank-deficient"<<endl;
//    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
    vector<int> vset;
    for (int i = 0; i <covSolver.eigenvalues().real().size() ; ++i) {
        if(covSolver.eigenvalues().real()(i)>ALPHA){
            vset.push_back(i);
        }
    }
    int rank=vset.size();
    //cout<<"rank: "<<rank<<endl;
    MatrixXd U;
    MatrixXd D;
    D.resize(rank,rank);
    U.resize(asize,rank);
    U.setZero();
    D.setZero();
    for (int i = 0; i <vset.size() ; ++i) {
        U.col(i)=covSolver.eigenvectors().real().col(vset[i]);
        D.block<1,1>(i,i)=covSolver.eigenvalues().real().row(vset[i]);
    }
    //cout<<"D matrix"<<endl<<D<<endl;
    MatrixXd Dinv=D.inverse();

    vector<pair<Eigen::MatrixXd,int>> infoVec;
    int hdim=0;
    for (int i = 1; i < vioRelativePoseEdges.size() ; ++i) {
        int row=vioRelativePoseEdges[i]->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * Dinv * (Ji * U).transpose());
        infoVec.emplace_back(covi.inverse(),hdim);
        cout<<"vioRelativePose "<<endl<<covi.inverse()<<endl;
        vioRelativePoseEdges[i]->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(covi.inverse()).matrixL().transpose();
        hdim+=row;
    }

    {
        int row = vioPosePriorEdge->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * Dinv * (Ji * U).transpose());
        infoVec.emplace_back(covi.inverse(),hdim);
        cout<<"vioPosePrior "<<endl<<covi.inverse()<<endl;
        vioPosePriorEdge->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(covi.inverse()).matrixL().transpose();
        hdim+=row;
    }

    {
        int row = vioVBPrior->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * Dinv * (Ji * U).transpose());
        infoVec.emplace_back(covi.inverse(),hdim);
        cout<<"vioVBPrior "<<endl<<covi.inverse()<<endl;
        vioVBPrior->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(covi.inverse()).matrixL().transpose();
        hdim+=row;
    }

    //test equality
    Eigen::MatrixXd X;
    X.resize(hdim,hdim);
    X.setZero();
    for (int i = 0; i <infoVec.size() ; ++i) {
        X.block(infoVec[i].second,infoVec[i].second,infoVec[i].first.rows(),infoVec[i].first.rows())+=infoVec[i].first;
    }
    //test KLD
    MatrixXd A=(Jr*U).transpose()*X*Jr*U;
    //cout<<"zero test"<<endl<<A-D<<endl;
    double a=(A*Dinv).trace();
    double b=log(A.determinant());
    double c=log(Dinv.determinant());
    double kld=0.5*(a-b-c-asize);
    cout<<"init KLD is "<<kld<<endl;



    int drop_size=factors2Sparsify.size();
    for (int i = 0; i <drop_size ; ++i) {
        delete factors2Sparsify[i];
    }
    factors2Sparsify.clear();


    double2vector();

}


void Estimator::problemSolve()
{
    if(!forwardProjectiontoSparsify.empty()){
        for (int i = 0; i <forwardProjectiontoSparsify.size() ; ++i) {
            delete forwardProjectiontoSparsify[i];
        }
    }
    forwardProjectiontoSparsify.clear();

    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership=ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);
    ceres::LossFunction *bigloss;
    bigloss = new ceres::CauchyLoss(0.5);

    for (int i = 0; i < ALL_BUF_SIZE; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

//    for (int i = Vo_SIZE - 1; i < ALL_BUF_SIZE - 1; i++)
    for (int i = 0; i < ALL_BUF_SIZE - 1; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor=new IMUFactor(pre_integrations[j]);
        imu_factor->setIndex(i,j);
        if(i == Vo_SIZE - 1){
            backwardIMUtoSparsify=imu_factor;
        }
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    int marglandmarks=0;
//    vector<int> stat;
    for (auto &idFeatures : f_manager.IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!f_manager.goodFeature(idFeatures))
            continue;

        ++feature_index;
 //       stat.push_back(idFeatures.used_num);

        int imu_i = idFeatures.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = idFeatures.idfeatures[0].point;
        for (auto &feature : idFeatures.idfeatures)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }

            Vector3d pts_j = feature.point;

            ProjectionFactor* f=new ProjectionFactor(pts_i, pts_j);
            f->setIndex(imu_i,imu_j,feature_index);

            int gap=imu_j-imu_i;
            if(imu_i==0  && gap==1 && marginalization_flag==MarginalizationFlag::MARGIN_OLD){
                forwardProjectiontoSparsify.push_back(f);
                MargPointIdx.push_back(feature_index);
                marglandmarks++;
            }

            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            f_m_cnt++;
        }
    }

//    double avg=0;
//    for (int i = 0; i <stat.size() ; ++i) {
//        avg+=stat[i];
//    }
//    cout<<"avgera is "<<avg/double(stat.size())<<endl;
    //cout<<marglandmarks<<" landmarks marginalized"<<endl;

    //add head pose factor#TODO BAD FACTOR!!!
    problem.AddResidualBlock(vioPosePriorEdge, bigloss, para_Pose[0]);

    //add VB prior factor
    problem.AddResidualBlock(vioVBPrior, bigloss, para_SpeedBias[Vo_SIZE - 1]);

    //add relative pose factor
    for (int i = 0; i < vioRelativePoseEdges.size() - 1 ; ++i) {
        int j=i+1;
        auto factor=vioRelativePoseEdges[j];
        problem.AddResidualBlock(factor,bigloss,para_Pose[i],para_Pose[j]);
    }
    for (int i = 0; i <vioRollPitchEdges.size() ; ++i) {
        auto factor=vioRollPitchEdges[i];
        int index=factor->index;
        problem.AddResidualBlock(factor,loss_function,para_Pose[index]);
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.BriefReport()<<endl;


    //update pseudo-measurement
    vioVBPrior->update(Vs[Vo_SIZE - 1],Bas[Vo_SIZE - 1],Bgs[Vo_SIZE - 1],para_SpeedBias[Vo_SIZE - 1]);
    vioPosePriorEdge->update(Ps[0],Rs[0],para_Pose[0]);
    for (int i = 0; i < vioRelativePoseEdges.size() - 1 ; ++i) {
        int j=i+1;
        auto factor=vioRelativePoseEdges[j];
        factor->update(Ps[i],Rs[i],Ps[j],Rs[j],para_Pose[i],para_Pose[j]);
    }
    for (int i = 0; i <vioRollPitchEdges.size() ; ++i) {
        auto factor=vioRollPitchEdges[i];
        int index=factor->index;
        factor->update(Rs[index],para_Pose[index]);
    }

}


void Estimator::MargForward() {
    int landamrk_size=MargPointIdx.size();
    int forward_dim= landamrk_size + 12;
    //T1 T0 landmarks
    std::unordered_map<double *,pair<int,int>> OrderMap;
    int idx=0;
    for (int i = 1; i >=0 ; i--) {
        OrderMap[para_Pose[i]]=make_pair(idx,6);
        idx+=6;
    }
    for (int i = 0; i <landamrk_size ; ++i) {
        OrderMap[para_Feature[MargPointIdx[i]]]=make_pair(idx,1);
        idx+=1;
    }
    
    Eigen::MatrixXd Lamda;
    Lamda.resize(forward_dim, forward_dim);
    Lamda.setZero();

    for (int i = 0; i < forwardProjectiontoSparsify.size() ; ++i) {
        auto factor=forwardProjectiontoSparsify[i];

        factor->EvaluateOnlyJacobians(para_Pose[0], para_Pose[1], para_Ex_Pose[0], para_Feature[MargPointIdx[i]]);

        MatrixXd infoMatrix=factor->sqrt_info.transpose()*factor->sqrt_info;
        //cout<<"info"<<endl<<infoMatrix<<endl;
        int imu_i=factor->imu_i;
        int imu_j=factor->imu_j;
        int f=factor->feature_idx;
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        ParamMap.push_back(para_Ex_Pose[0]);
        ParamMap.push_back(para_Feature[f]);
        for (int j = 0; j <ParamMap.size() ; ++j) {
            if(OrderMap.find(ParamMap[j])==OrderMap.end())continue;
            auto v_j=OrderMap[ParamMap[j]];
            int idxj=v_j.first;
            int dimj=v_j.second;
            MatrixXd JtW = factor->jacobians[j].transpose() * infoMatrix;
            //cout<<"j"<<endl<<jacobians[j]<<endl;
            for (int k = j; k <ParamMap.size() ; ++k) {
                if(OrderMap.find(ParamMap[k])==OrderMap.end())continue;
                auto v_k=OrderMap[ParamMap[k]];
                int idxk=v_k.first;
                int dimk=v_k.second;
                Eigen::MatrixXd Hessian=JtW*factor->jacobians[k];
                Lamda.block(idxj, idxk, dimj, dimk).noalias()+=Hessian;
                if(j!=k){
                    Lamda.block(idxk, idxj, dimk, dimj).noalias()+=Hessian.transpose();
                }
            }
        }
    }
    {
        vioPosePriorEdge->EvaluateOnlyJacobians(para_Pose[0]);
        MatrixXd infoMatrix=vioPosePriorEdge->sqrt_info.transpose()*vioPosePriorEdge->sqrt_info;
        auto v_j=OrderMap[para_Pose[0]];
        int idxj=v_j.first;
        int dimj=v_j.second;
        Lamda.block(idxj, idxj, dimj, dimj).noalias()+=vioPosePriorEdge->jacobians[0].transpose()*
                infoMatrix*vioPosePriorEdge->jacobians[0];
    }
    {
        vioRelativePoseEdges[1]->EvaluateOnlyJacobians(para_Pose[0], para_Pose[1]);
        MatrixXd infoMatrix= vioRelativePoseEdges[1]->sqrt_info.transpose() * vioRelativePoseEdges[1]->sqrt_info;
        auto jacobians=vioRelativePoseEdges[1]->jacobians;
       int imu_i=vioRelativePoseEdges[1]->imu_i;
       int imu_j=vioRelativePoseEdges[1]->imu_j;
       assert(imu_i==0 &&  imu_j==1);
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        for (int j = 0; j <ParamMap.size() ; ++j) {
            auto v_j=OrderMap[ParamMap[j]];
            int idxj=v_j.first;
            int dimj=v_j.second;
            MatrixXd JtW = jacobians[j].transpose() * infoMatrix;
            for (int k = j; k <ParamMap.size() ; ++k) {
                auto v_k=OrderMap[ParamMap[k]];
                int idxk=v_k.first;
                int dimk=v_k.second;
                Eigen::MatrixXd Hessian=JtW*jacobians[k];
                Lamda.block(idxj, idxk, dimj, dimk).noalias()+=Hessian;
                if(j!=k){
                    Lamda.block(idxk, idxj, dimk, dimj).noalias()+=Hessian.transpose();
                }
            }
        }
    }

    Eigen::MatrixXd Lamda_rr=Lamda.block(0, 0, 6, 6);
    Eigen::MatrixXd Lamda_mm=Lamda.block(6, 6, landamrk_size + 6, landamrk_size + 6);

    Eigen::MatrixXd Lamda_rp=Lamda.block(0,0,12,12);
    Vector3d Psj,Psi;
    Psi<<para_Pose[0][0], para_Pose[0][1], para_Pose[0][2];
    Psj<<para_Pose[1][0], para_Pose[1][1], para_Pose[1][2];
    Quaterniond Qi(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5]);
    Quaterniond Qj(para_Pose[1][6], para_Pose[1][3], para_Pose[1][4], para_Pose[1][5]);
    Vector3d tij=Psj-Psi;
    Matrix3d Rij=(Qi.inverse()*Qj).toRotationMatrix();
    RelativePoseFactor *pgRaltivePoseFactor=new RelativePoseFactor(tij,Rij);
    pgRaltivePoseFactor->EvaluateOnlyJacobians(para_Pose[0],para_Pose[1]);
    Matrix<double,6,12> J;
    J.topLeftCorner<6,6>()=pgRaltivePoseFactor->jacobians[0];
    J.bottomRightCorner<6,6>()=pgRaltivePoseFactor->jacobians[1];
    MatrixXd Jpinv=Utility::pseudoInverse<MatrixXd>(J,1e-8);
    MatrixXd rpOmega=Jpinv.transpose()*Lamda_rp*Jpinv;
    MatrixXd rpCov=(rpOmega).inverse();
    pgRaltivePoseFactor->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(rpOmega).matrixL().transpose();


    //send to pose graph
    CombinedFactors *cmbfactors=new CombinedFactors();
    cmbfactors->relativePoseFactor=pgRaltivePoseFactor;
    if(!vioRollPitchEdges.empty()){
        if(vioRollPitchEdges[0]->index==0){
            cmbfactors->rollPitchFactor=vioRollPitchEdges[0];
            cmbfactors->covAbs= (vioRollPitchEdges[0]->sqrt_info.transpose()*vioRollPitchEdges[0]->sqrt_info).inverse();
        }else
            cmbfactors->rollPitchFactor= nullptr;
    }

    cmbfactors->vio_index=PoseGraphFactorCount;
    cmbfactors->distance=pgRaltivePoseFactor->delta_t.norm();
    cmbfactors->covRel= rpCov;
    cmbfactors->ts=Headers[0];
    cmbfactors->R=Rs[0];
    cmbfactors->t=Ps[0];
    PoseGraphFactorCount++;

    m_pose_graph_buf.lock();
    pose_graph_factors_buf.push(cmbfactors);
    m_pose_graph_buf.unlock();


    Eigen::MatrixXd Lamda_mm_inv = Lamda_mm.fullPivLu().solve(Eigen::MatrixXd::Identity(landamrk_size + 6, landamrk_size + 6));
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, 6, 6, landamrk_size + 6);
    auto Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();

    //keep pose for the first pose
    Vector3d P1;
    P1<<para_Pose[1][0],para_Pose[1][1], para_Pose[1][2];
    Quaterniond Q1(para_Pose[1][6],para_Pose[1][3], para_Pose[1][4], para_Pose[1][5]);
    SE3PriorFactor* se3PriorFactor=new SE3PriorFactor(P1,Q1);
    se3PriorFactor->EvaluateOnlyJacobians(para_Pose[1]);
    MatrixXd infoMatrix=se3PriorFactor->sqrt_info.transpose()*se3PriorFactor->sqrt_info;
    Matrix<double,6,6> Jr=se3PriorFactor->jacobians[0];

//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
//    if(Jqr.rank()<Jr.rows()){
//        cerr<<"Jacobian of forward recovered factor is rank-deficient"<<endl;
//    }

    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(Lamda_prior);
    qr.setThreshold(eps);
    MatrixXd covi;
    if(qr.rank()==Lamda_prior.cols()){
        Eigen::MatrixXd cov=qr.solve(Eigen::MatrixXd::Identity(6,6));
        covi=Jr*cov*Jr.transpose();
    }else{
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
        vector<int> vset;
        for (int i = 0; i <covSolver.eigenvalues().real().size() ; ++i) {
            if(covSolver.eigenvalues().real()(i)>ALPHA){
                vset.push_back(i);
            }
        }
        int rank=vset.size();
        MatrixXd U;
        MatrixXd D;
        D.resize(rank,rank);
        U.resize(6,rank);
        U.setZero();
        D.setZero();
        for (int i = 0; i <vset.size() ; ++i) {
            U.col(i)=covSolver.eigenvectors().real().col(vset[i]);
            D.block<1,1>(i,i)=covSolver.eigenvalues().real().row(vset[i]);
        }
        MatrixXd Dinv=D.inverse();
        covi=Jr*U*Dinv*(Jr*U).transpose();
    }

    Eigen::MatrixXd X;
    X.resize(6,6);
    X=covi.inverse();
    if(qr.rank()==Lamda_prior.cols()){
        auto phi=Jr.transpose()*X*Jr;
        Eigen::MatrixXd cov=qr.solve(Eigen::MatrixXd::Identity(6,6));

        double a=(phi*cov).trace();
        double b=log(phi.determinant());
        double c=log(cov.determinant());
        double kld=0.5*(a-b-c-6);
        //cout<<"forward KLD is "<<kld<<endl;
    }


    //cout<<"se3PriorFactor "<<endl;
    se3PriorFactor->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(covi.inverse()).matrixL().transpose();

    forwardPosePriorEdgeToAdd=se3PriorFactor;
}

void Estimator::MargBackward(){

    //      T1 VB1 T0 VB0
    //order  0  6  15  21 30
    std::unordered_map<double *,pair<int,int>> OrderMap;
    int idx=0;
    for (int i = Vo_SIZE; i >= Vo_SIZE - 1; i--) {

        OrderMap[para_Pose[i]]=make_pair(idx,6);
        idx+=6;
        OrderMap[para_SpeedBias[i]]=make_pair(idx,9);
        idx+=9;
    }

    Eigen::MatrixXd Lamda;
    Lamda.resize(30, 30);
    Lamda.setZero();

    {
        vioVBPrior->EvaluateOnlyJacobians(para_SpeedBias[Vo_SIZE - 1]);
        vector<MatrixXd> jacobians=vioVBPrior->jacobians;
        MatrixXd infoMatrix= vioVBPrior->sqrt_info.transpose() * vioVBPrior->sqrt_info;
        auto v_j=OrderMap[para_SpeedBias[Vo_SIZE - 1]];
        int idxj=v_j.first;
        int dimj=v_j.second;
        Lamda.block(idxj, idxj, dimj, dimj).noalias()+=jacobians[0].transpose()*infoMatrix*jacobians[0];
    }

    {
        auto imufactor=backwardIMUtoSparsify;
        int imu_i=imufactor->imu_i;
        int imu_j=imufactor->imu_j;
        assert(imu_i == Vo_SIZE - 1 && imu_j == Vo_SIZE);
        imufactor->Evaluate(para_Pose[Vo_SIZE - 1], para_SpeedBias[Vo_SIZE - 1],
                            para_Pose[Vo_SIZE], para_SpeedBias[Vo_SIZE]);
        vector<MatrixXd> jacobiansI=imufactor->jacobians;
        MatrixXd omegaI=imufactor->sqrt_info.transpose()*imufactor->sqrt_info;
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_SpeedBias[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        ParamMap.push_back(para_SpeedBias[imu_j]);
        for (int j = 0; j <ParamMap.size() ; ++j) {
            auto v_j=OrderMap[ParamMap[j]];
            int index_j=v_j.first;
            int dim_j=v_j.second;
            MatrixXd JtW = jacobiansI[j].transpose() * omegaI;
            for (int k = j; k <ParamMap.size() ; ++k) {
                auto v_k=OrderMap[ParamMap[k]];
                int index_k=v_k.first;
                int dim_k=v_k.second;
                MatrixXd hessian = JtW * jacobiansI[k];
                Lamda.block(index_j, index_k, dim_j, dim_k).noalias() += hessian;
                if (j != k) {
                    Lamda.block(index_k, index_j, dim_k, dim_j).noalias() += hessian.transpose();
                }
            }
        }
    }
    Eigen::MatrixXd Lamda_rr=Lamda.block(0, 0, 21, 21);
    Eigen::MatrixXd Lamda_mm=Lamda.block(21, 21, 9, 9);
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, 21, 21, 9);

    Eigen::MatrixXd Lamda_mm_inv = Lamda_mm.fullPivLu().solve(Eigen::MatrixXd::Identity(9, 9));

    MatrixXd Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();


    //recovery information
//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(Lamda_prior);
//    cout << "backward information has rank " << qr.rank() << " with rows and cols " << Lamda_prior.rows() << endl;


    Vector3d Psj,Psi;
    Psi<<para_Pose[Vo_SIZE - 1][0], para_Pose[Vo_SIZE - 1][1], para_Pose[Vo_SIZE - 1][2];
    Psj<<para_Pose[Vo_SIZE][0], para_Pose[Vo_SIZE][1], para_Pose[Vo_SIZE][2];
    Quaterniond Qi(para_Pose[Vo_SIZE - 1][6], para_Pose[Vo_SIZE - 1][3], para_Pose[Vo_SIZE - 1][4], para_Pose[Vo_SIZE - 1][5]);
    Quaterniond Qj(para_Pose[Vo_SIZE][6], para_Pose[Vo_SIZE][3], para_Pose[Vo_SIZE][4], para_Pose[Vo_SIZE][5]);
    Vector3d tij=Psj-Psi;
    Matrix3d Rij=(Qi.inverse()*Qj).toRotationMatrix();

    RelativePoseFactor* relativePoseFactor=new RelativePoseFactor(tij,Rij);
    relativePoseFactor->EvaluateOnlyJacobians(para_Pose[Vo_SIZE - 1], para_Pose[Vo_SIZE]);

    //keep vb
    Eigen::VectorXd vb(9);
    vb << para_SpeedBias[Vo_SIZE][0], para_SpeedBias[Vo_SIZE][1], para_SpeedBias[Vo_SIZE][2],
            para_SpeedBias[Vo_SIZE][3], para_SpeedBias[Vo_SIZE][4], para_SpeedBias[Vo_SIZE][5],
            para_SpeedBias[Vo_SIZE][6], para_SpeedBias[Vo_SIZE][7], para_SpeedBias[Vo_SIZE][8];
    Linear9Factor* speedBiasPrior=new Linear9Factor(vb);
    speedBiasPrior->EvaluateOnlyJacobians(para_SpeedBias[Vo_SIZE]);

//   //keep rollpitch for the last pose
    Quaterniond Qw(para_Pose[Vo_SIZE - 1][6], para_Pose[Vo_SIZE - 1][3], para_Pose[Vo_SIZE - 1][4], para_Pose[Vo_SIZE - 1][5]);
    RollPitchFactor *rollPitchFactor=new RollPitchFactor(Qw);
    rollPitchFactor->EvaluateOnlyJacobians(para_Pose[Vo_SIZE - 1]);

    YawFactor* yawFactor=new YawFactor(Qw);
    yawFactor->EvaluateOnlyJacobians(para_Pose[Vo_SIZE - 1]);

    //order t1 R1 VB1 t0 R0
   //col_id:0  3   6  15 18 21
    MatrixXd Jr;
    Jr.resize(21,21);
    Jr.setZero();
    Jr.block(0,15,6,6).noalias()+=relativePoseFactor->jacobians[0];
    Jr.block(0,0,6,6).noalias()+=relativePoseFactor->jacobians[1];
    Jr.block(6,6,9,9).noalias()+=speedBiasPrior->jacobians[0];
    Jr.block(15,15,2,6).noalias()+=rollPitchFactor->jacobians[0];
    Jr.block(17,15,3,3).noalias()+=Matrix3d::Identity();//absolute position
    Jr.block(20,15,1,6).noalias()+=yawFactor->jacobians[0];



//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
//    if(Jqr.rank()<Jr.rows()){
//        cerr<<"Jacobian of backward recovered factor is rank-deficient"<<endl;
//    }

    MatrixXd Jrp=Jr.block(0,0,6,21);
    MatrixXd Jvb=Jr.block(6,0,9,21);
    MatrixXd Jgv=Jr.block(15,0,2,21);
    MatrixXd Jabs=Jr.block(17,0,3,21);
    MatrixXd Jyaw=Jr.block(20,0,1,21);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
    vector<int> vset;
    for (int i = 0; i <covSolver.eigenvalues().real().size() ; ++i) {
        if(covSolver.eigenvalues().real()(i)>ALPHA){
            vset.push_back(i);
        }
    }
    int rank=vset.size();
    MatrixXd U;
    MatrixXd D;
    D.resize(rank,rank);
    U.resize(21,rank);
    U.setZero();
    D.setZero();
    for (int i = 0; i <vset.size() ; ++i) {
        U.col(i)=covSolver.eigenvectors().real().col(vset[i]);
        D.block<1,1>(i,i)=covSolver.eigenvalues().real().row(vset[i]);
    }
    auto Dinv=D.inverse();

    vector<pair<Eigen::MatrixXd,int>> infoVec;
    MatrixXd RPinfo=Jrp*U*Dinv*(Jrp*U).transpose();
    infoVec.emplace_back(RPinfo.inverse(),0);
    //cout<<"relativePoseFactor "<<endl<<RPinfo.inverse()<<endl;
    relativePoseFactor->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(RPinfo.inverse()).matrixL().transpose();

    MatrixXd VBinfo=Jvb*U*Dinv*(Jvb*U).transpose();
    infoVec.emplace_back(VBinfo.inverse(),6);
    //cout<<"speedBiasPrior "<<endl<<VBinfo.inverse()<<endl;
    speedBiasPrior->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(VBinfo.inverse()).matrixL().transpose();

    MatrixXd GVinfo=Jgv*U*Dinv*(Jgv*U).transpose(); //roll and pitch factor,aligned with gravity

    //test
    //cout<<"rollPitchFactor "<<endl<<GVinfo.inverse()<<endl;
    infoVec.emplace_back(GVinfo.inverse(),15);
    rollPitchFactor->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(GVinfo.inverse()).matrixL().transpose();
    rollPitchFactor->setIndex(Vo_SIZE - 1);

    infoVec.emplace_back((Jabs*U*Dinv*(Jabs*U).transpose()).inverse(),17);
    infoVec.emplace_back((Jyaw*U*Dinv*(Jyaw*U).transpose()).inverse(),20);

    //TODO TEST edgeRollPitch information with before one
    Eigen::MatrixXd X;
    X.resize(21,21);
    X.setZero();
    for (int i = 0; i <infoVec.size() ; ++i) {
        X.block(infoVec[i].second,infoVec[i].second,infoVec[i].first.rows(),infoVec[i].first.rows())+=infoVec[i].first;
    }
    MatrixXd A=(Jr*U).transpose()*X*Jr*U;
    //cout<<"zero test"<<endl<<A-D<<endl;
    double a=(A*Dinv).trace();
    double b=log(A.determinant());
    double c=log(Dinv.determinant());
    double kld=0.5*(a-b-c-21);
    cout<<"backward KLD is "<<kld<<endl;

    vioRollPitchEdges.push_back(rollPitchFactor);
    backwardVBEdgeToAdd=speedBiasPrior;
    backwardRelativePoseEdgeToAdd=relativePoseFactor;
}

void Estimator::backendOptimization()
{

    if(solver_flag==INITIAL_STRUCTURE){
        vector2double();
        initFactorGraph();
        solver_flag=NON_LINEAR;
        cout<<"pose graph initialized"<<endl;
    }
    if(solver_flag==NON_LINEAR){
        vector2double();
        problemSolve();
        double2vector();
        if (marginalization_flag == MARGIN_OLD) {
            MargForward();
            MargBackward();
        }
        MargPointIdx.clear();
        features2Marg.clear();
    }

}


void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == ALL_BUF_SIZE-1)
        {
            for (int i = 0; i < ALL_BUF_SIZE-1; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[ALL_BUF_SIZE-1] = Headers[ALL_BUF_SIZE - 2];
            Ps[ALL_BUF_SIZE-1] = Ps[ALL_BUF_SIZE - 2];
            Vs[ALL_BUF_SIZE-1] = Vs[ALL_BUF_SIZE - 2];
            Rs[ALL_BUF_SIZE-1] = Rs[ALL_BUF_SIZE - 2];
            Bas[ALL_BUF_SIZE-1] = Bas[ALL_BUF_SIZE - 2];
            Bgs[ALL_BUF_SIZE-1] = Bgs[ALL_BUF_SIZE - 2];

            delete pre_integrations[ALL_BUF_SIZE-1];
            pre_integrations[ALL_BUF_SIZE-1] = new IntegrationBase{acc_0, gyr_0, Bas[ALL_BUF_SIZE-1], Bgs[ALL_BUF_SIZE-1]};

            dt_buf[ALL_BUF_SIZE-1].clear();
            linear_acceleration_buf[ALL_BUF_SIZE-1].clear();
            angular_velocity_buf[ALL_BUF_SIZE-1].clear();

            if(solver_flag==Estimator::NON_LINEAR){
                for (int i = 1; i < Vo_SIZE ; ++i) {
                    vioRelativePoseEdges[i]->shift();
                }
                for (int i = 1; i < Vo_SIZE - 1 ; ++i) {
                    std::swap(vioRelativePoseEdges[i], vioRelativePoseEdges[i + 1]);// vioRelativePoseEdges[0] is nullptr
                }

                int drop_size=forwardProjectiontoSparsify.size();
                for (int i = 0; i <drop_size ; ++i) {
                    delete forwardProjectiontoSparsify[i];
                }
                forwardProjectiontoSparsify.clear();

                for (auto iter=vioRollPitchEdges.begin();iter!=vioRollPitchEdges.end();) {
                    (*iter)->shift();
                    if((*iter)->index<0){
                        //delete (*iter); //used in pose graph
                        iter=vioRollPitchEdges.erase(iter);
                    }else
                        iter++;
                }

                //delete vioRelativePoseEdges[Vo_SIZE - 1];//used in pose graph

                backwardRelativePoseEdgeToAdd->setIndex(Vo_SIZE - 2, Vo_SIZE - 1);
                vioRelativePoseEdges[Vo_SIZE - 1]=backwardRelativePoseEdgeToAdd;
                delete vioPosePriorEdge;
                forwardPosePriorEdgeToAdd->setIndex(0);
                vioPosePriorEdge=forwardPosePriorEdgeToAdd;
                delete vioVBPrior;
                backwardVBEdgeToAdd->setIndex(Vo_SIZE - 1);
                vioVBPrior=backwardVBEdgeToAdd;

            }


            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == ALL_BUF_SIZE-1)
        {
            double t_1 = Headers[frame_count];
            //keep IMU data for the first frame in Vio_SZIE
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count-1] = Headers[frame_count];
            Ps[frame_count-1] = Ps[frame_count ];
            Vs[frame_count-1] = Vs[frame_count ];
            Rs[frame_count-1] = Rs[frame_count ];
            Bas[frame_count-1] = Bas[frame_count];
            Bgs[frame_count-1] = Bgs[frame_count ];

            delete pre_integrations[ALL_BUF_SIZE-1];
            pre_integrations[ALL_BUF_SIZE-1] = new IntegrationBase{acc_0, gyr_0, Bas[ALL_BUF_SIZE-1], Bgs[ALL_BUF_SIZE-1]};

            dt_buf[ALL_BUF_SIZE-1].clear();
            linear_acceleration_buf[ALL_BUF_SIZE-1].clear();
            angular_velocity_buf[ALL_BUF_SIZE-1].clear();

            slideWindowNew();
        }
    }
}


void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}