#include "estimator.h"
#include <ostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

MatrixXd extractInfo(MatrixXd &info){
    double eps=1e-8;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Solver(info);
    Eigen::MatrixXd Omega = Solver.eigenvectors() *
                            Eigen::VectorXd((Solver.eigenvalues().array() > eps).select(Solver.eigenvalues().array().inverse(), 0)).asDiagonal() *
                            Solver.eigenvectors().transpose();
    return Eigen::LLT<Eigen::MatrixXd>(Omega).matrixL().transpose();
}

Estimator::Estimator() : f_manager{Rs}
{
    // ROS_INFO("init begins");

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
        // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
        //     << " ric: " << ric[i] << endl;
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
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

    failure_occur = 0;
    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
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
        //if(solver_flag != NON_LINEAR)
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
    //ROS_DEBUG("new image coming ------------------------------------------");
    // cout << "Adding IDsfeatures points: " << image.size()<<endl;
    if (f_manager.addFeatureAndCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_NEW;
    //cout<<"frame_count "<<frame_count<<endl;
    //cout << "number of IDsfeatures: " << f_manager.getFeatureCount()<<endl;
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
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                            //    << calib_ric);
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
                // cout << "1 initialStructure" << endl;
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
        //ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            // ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            cout<<"detected failure"<<endl;
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        //ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
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
        //ROS_DEBUG("solve g failed!");
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
    //ROS_DEBUG_STREAM("g0     " << g.transpose());
    //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

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

    // relative info between two loop frame
    if(relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        m_loop_buf.lock();
        Eigen::Matrix<double, 8, 1 > loop_info;
        loop_info << relo_relative_t.x(), relo_relative_t.y(), relo_relative_t.z(),
                relo_relative_q.w(), relo_relative_q.x(), relo_relative_q.y(), relo_relative_q.z(),
                relo_relative_yaw;
        loop_buf.push(make_pair(relo_frame_local_index,loop_info));
        m_loop_buf.unlock();
        relocalization_info = 0;

    }

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
        failed= true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        cout<<" big z translation"<<endl;
        failed= true;
    }
    Matrix3d tmp_R = Rs[ALL_BUF_SIZE-1];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        cout<<" big delta_angle "<<endl;
        failed= true;
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
        return true;
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
            //ROS_DEBUG("fix extinsic param");
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

    int f_m_cnt = 0;
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
            f_m_cnt++;
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = SOLVER_TIME;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;

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


    MatrixXd Lamda;
    Lamda.resize(Vo_SIZE * 15, Vo_SIZE * 15);
    Lamda.setZero();
    assert(factors2Sparsify.size() == Vo_SIZE - 1);
    for (int i = 0; i <factors2Sparsify.size() ; ++i) {
        IMUFactor* factor= factors2Sparsify[i];
        int imu_i=factor->imu_i;
        int imu_j=factor->imu_j;

        factor->Evaluate(para_Pose[imu_i],para_SpeedBias[imu_i],para_Pose[imu_j],para_SpeedBias[imu_j]);
        vector<MatrixXd> jacobians=factor->jacobians;
        MatrixXd robustInfo=factor->sqrt_info.transpose()*factor->sqrt_info;
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_SpeedBias[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        ParamMap.push_back(para_SpeedBias[imu_j]);
        
        for (int j = 0; j <ParamMap.size() ; ++j) {
            auto v_j=OrderMap[ParamMap[j]];
            int index_j=v_j.first;
            int dim_j=v_j.second;
            MatrixXd JtW = jacobians[j].transpose() * robustInfo;
            for (int k = j; k <ParamMap.size() ; ++k) {
                auto v_k=OrderMap[ParamMap[k]];
                int index_k=v_k.first;
                int dim_k=v_k.second;
                MatrixXd hessian = JtW * jacobians[k];
                Lamda.block(index_j, index_k, dim_j, dim_k).noalias() += hessian;
                if (j != k) {
                    Lamda.block(index_k, index_j, dim_k, dim_j).noalias() += hessian.transpose();

                }
            }
        }
    }


    int t_psvb_dim= Vo_SIZE * 6 + 9;
    int t_vbs_dim= (Vo_SIZE - 1) * 9;

    double eps=1e-8;
    Eigen::MatrixXd Lamda_rr=Lamda.block(0, 0, t_psvb_dim, t_psvb_dim);
    Eigen::MatrixXd Lamda_mm=Lamda.block(t_psvb_dim, t_psvb_dim, t_vbs_dim, t_vbs_dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> initSolver(Lamda_mm);
    Eigen::MatrixXd Lamda_mm_inv = initSolver.eigenvectors() *
            Eigen::VectorXd((initSolver.eigenvalues().array() > eps).select(initSolver.eigenvalues().array().inverse(), 0)).asDiagonal()
            *initSolver.eigenvectors().transpose();
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, t_psvb_dim, t_psvb_dim, t_vbs_dim);
    auto Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();
    assert(Lamda_prior.rows() == Vo_SIZE * 6 + 9);

//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(Lamda_prior);
//    qr.setThreshold(eps);
//    cout << "Initial Information  has rank " << qr.rank() << " with rows and cols " << Lamda_prior.rows() << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
    Eigen::VectorXd S = Eigen::VectorXd((covSolver.eigenvalues().array() > eps).select(covSolver.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (covSolver.eigenvalues().array() > eps).select(covSolver.eigenvalues().array().inverse(), 0));

    MatrixXd U=covSolver.eigenvectors();
    MatrixXd cov=S_inv.asDiagonal();

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


    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
    if(Jqr.rank()<Jr.rows()){
        cerr<<"Jacobian of initial recovered factor is rank-deficient"<<endl;
    }

    //set information for each edge

    int hdim=0;
    for (int i = 1; i < vioRelativePoseEdges.size() ; ++i) {
        int row=vioRelativePoseEdges[i]->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * cov * (Ji * U).transpose());
        vioRelativePoseEdges[i]->sqrt_info=extractInfo(covi);
        hdim+=row;
    }

    {
        int row = vioPosePriorEdge->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * cov * (Ji * U).transpose());
        vioPosePriorEdge->sqrt_info=extractInfo(covi);
        hdim+=row;
    }

    {
        int row = vioVBPrior->num_residuals();
        MatrixXd Ji=Jr.block(hdim,0,row, 6 * Vo_SIZE + 9);
        MatrixXd covi=(Ji * U * cov * (Ji * U).transpose());
        vioVBPrior->sqrt_info=extractInfo(covi);
        hdim+=row;
    }

    int drop_size=factors2Sparsify.size();
    for (int i = 0; i <drop_size ; ++i) {
        delete factors2Sparsify[i];
    }
    factors2Sparsify.clear();


    double2vector();
//test KLD
//    double a=(A_nfr*cov).trace();
//    double b= log(Lamda_rr.determinant()) - log(A_nfr.determinant());
//    double kld=0.5*(a + b - Lamda_rr.rows());
//    cout<<"init KLD is "<<kld<<endl;
}


void Estimator::problemSolve()
{
    bool checkNullSpace=false;
    vector<ceres::CostFunction*> factors;
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
    ceres::LossFunction *bigloss = new ceres::CauchyLoss(0.3);
    loss_function = new ceres::CauchyLoss(1.0);

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
            //ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

    for (int i = Vo_SIZE - 1; i < ALL_BUF_SIZE - 1; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor =new IMUFactor(pre_integrations[j]);
        if(checkNullSpace){
            factors.push_back(imu_factor);
        }
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

            if(checkNullSpace){
                factors.push_back(f);
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

    //add head pose factor
    problem.AddResidualBlock(vioPosePriorEdge, bigloss, para_Pose[0]);

    //add VB prior factor
    problem.AddResidualBlock(vioVBPrior, loss_function, para_SpeedBias[Vo_SIZE - 1]);

    if(checkNullSpace){
        factors.push_back(vioPosePriorEdge);
        factors.push_back(vioVBPrior);

    }
    //add relative pose factor
    for (int i = 0; i < vioRelativePoseEdges.size() - 1 ; ++i) {
        int j=i+1;
        auto factor=vioRelativePoseEdges[j];
        problem.AddResidualBlock(factor,bigloss,para_Pose[i],para_Pose[j]);
        if(checkNullSpace){
            factors.push_back(factor);
        }
    }
    ceres::ResidualBlockId rid;
    for (int i = 0; i <vioRollPitchEdges.size() ; ++i) {
        auto factor=vioRollPitchEdges[i];
        int index=factor->index;
        rid=problem.AddResidualBlock(factor,NULL,para_Pose[index]);
        if(checkNullSpace){
            factors.push_back(factor);
        }
    }
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &idFeatures : f_manager.IDsfeatures)
        {
            idFeatures.used_num = idFeatures.idfeatures.size();
            if (!f_manager.goodFeature(idFeatures))
                continue;
            ++feature_index;
            int start = idFeatures.start_frame;
            if(start <= relo_frame_local_index)
            {
                while((int)match_points[retrive_feature_index].z() < idFeatures.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == idFeatures.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = idFeatures.idfeatures[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }

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
    //check nullspace

//    std::unordered_map<double *,pair<int,int>> OrderMap;
//    int idx=0;
//    for (int i = 0; i<ALL_BUF_SIZE ; i++) {
//        OrderMap[para_Pose[i]]=make_pair(idx,6);
//        idx+=6;
//        OrderMap[para_SpeedBias[i]]=make_pair(idx,9);
//        idx+=9;
//    }
//    for (int i = 0; i <feature_index ; ++i) {
//        OrderMap[para_Feature[i]]=make_pair(idx,1);
//        idx+=1;
//    }
//    int full_size=ALL_BUF_SIZE*15+feature_index;
//    MatrixXd H;
//    H.resize(full_size,full_size);


    //cout<<summary.BriefReport()<<endl;
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

   // cout<<"Lamda"<<endl<<Lamda<<endl;
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

    double eps=1e-8;
//    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> Solver(rpOmega);
//    cout<<"test rp omega "<<endl;
//    cerr<<Solver.eigenvalues().transpose()<<endl;

    pgRaltivePoseFactor->sqrt_info=Eigen::LLT<Eigen::MatrixXd>(rpOmega).matrixL().transpose();
    pgRaltivePoseFactor->setGlobalIndex(0);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> initSolver(Lamda_mm);
    Eigen::MatrixXd Lamda_mm_inv = initSolver.eigenvectors() *
            Eigen::VectorXd((initSolver.eigenvalues().array() > eps).select(initSolver.eigenvalues().array().inverse(), 0)).asDiagonal() *
                                   initSolver.eigenvectors().transpose();
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, 6, 6, landamrk_size + 6);
    auto Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();

//    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> qr(Lamda_prior);
//    qr.setThreshold(eps);
//    cout << "Forward Information  has rank " << qr.rank() << " with rows and cols " << Lamda_prior.rows() << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
    Eigen::MatrixXd cov = covSolver.eigenvectors() *
                            Eigen::VectorXd((covSolver.eigenvalues().array() > eps).select(covSolver.eigenvalues().array().inverse(), 0)).asDiagonal() *
            covSolver.eigenvectors().transpose();

    //keep pose for the first pose
    Vector3d P1;
    P1<<para_Pose[1][0],para_Pose[1][1], para_Pose[1][2];
    Quaterniond Q1(para_Pose[1][6],para_Pose[1][3], para_Pose[1][4], para_Pose[1][5]);
    SE3PriorFactor* se3PriorFactor=new SE3PriorFactor(P1,Q1);
    se3PriorFactor->EvaluateOnlyJacobians(para_Pose[1]);
    MatrixXd infoMatrix=se3PriorFactor->sqrt_info.transpose()*se3PriorFactor->sqrt_info;
    Matrix<double,6,6> Jr=se3PriorFactor->jacobians[0];

    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
    if(Jqr.rank()<Jr.rows()){
        cerr<<"Jacobian of forward recovered factor is rank-deficient"<<endl;
    }

    MatrixXd info=Jr*cov*Jr.transpose();
    se3PriorFactor->sqrt_info=extractInfo(info);

   // cout<<"recovered forward pose info "<<endl<<Omega<<endl;
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
        backwardIMUtoSparsify->Evaluate(para_Pose[Vo_SIZE - 1], para_SpeedBias[Vo_SIZE - 1],
                                        para_Pose[Vo_SIZE], para_SpeedBias[Vo_SIZE]);
        vector<MatrixXd> jacobians=backwardIMUtoSparsify->jacobians;
        MatrixXd infoMatrix=backwardIMUtoSparsify->sqrt_info.transpose()*backwardIMUtoSparsify->sqrt_info;
        int imu_i=backwardIMUtoSparsify->imu_i;
        int imu_j=backwardIMUtoSparsify->imu_j;
        assert(imu_i == Vo_SIZE - 1 && imu_j == Vo_SIZE);
        vector<double *> ParamMap;
        ParamMap.push_back(para_Pose[imu_i]);
        ParamMap.push_back(para_SpeedBias[imu_i]);
        ParamMap.push_back(para_Pose[imu_j]);
        ParamMap.push_back(para_SpeedBias[imu_j]);
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
    Eigen::MatrixXd Lamda_rr=Lamda.block(0, 0, 21, 21);
    Eigen::MatrixXd Lamda_mm=Lamda.block(21, 21, 9, 9);
    Eigen::MatrixXd Lamda_rm=Lamda.block(0, 21, 21, 9);

    double eps=1e-8;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> initSolver(Lamda_mm);
    Eigen::MatrixXd Lamda_mm_inv = initSolver.eigenvectors() * Eigen::VectorXd(
            (initSolver.eigenvalues().array() > eps).select(initSolver.eigenvalues().array().inverse(), 0)).asDiagonal() *
                                   initSolver.eigenvectors().transpose();

    auto Lamda_prior= Lamda_rr - Lamda_rm * Lamda_mm_inv * Lamda_rm.transpose();


    //cout<<Lamda_prior<<endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSolver(Lamda_prior);
    Eigen::VectorXd S = Eigen::VectorXd((covSolver.eigenvalues().array() > eps).select(covSolver.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (covSolver.eigenvalues().array() > eps).select(covSolver.eigenvalues().array().inverse(), 0));
    MatrixXd U=covSolver.eigenvectors();
    MatrixXd cov=S_inv.asDiagonal();

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


    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> Jqr(Jr);
    if(Jqr.rank()<Jr.rows()){
        cerr<<"Jacobian of backward recovered factor is rank-deficient"<<endl;
    }

    MatrixXd Jrp=Jr.block(0,0,6,21);
    MatrixXd Jvb=Jr.block(6,0,9,21);
    MatrixXd Jgv=Jr.block(15,0,2,21);

    MatrixXd RPinfo=Jrp*U*cov*(Jrp*U).transpose();
    relativePoseFactor->sqrt_info=extractInfo(RPinfo);

    MatrixXd VBinfo=Jvb*U*cov*(Jvb*U).transpose();
    speedBiasPrior->sqrt_info=extractInfo(VBinfo);

    MatrixXd GVinfo=Jgv*U*cov*(Jgv*U).transpose(); //roll and pitch factor,aligned with gravity
    rollPitchFactor->sqrt_info=extractInfo(GVinfo);
    rollPitchFactor->setIndex(Vo_SIZE - 1);

    //TODO TEST edgeRollPitch information with before one

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
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < ALL_BUF_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i])
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}