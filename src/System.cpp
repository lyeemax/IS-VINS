#include "System.h"

#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace pangolin;

System::System(string sConfig_file_)
    :bStart_backend(true)
{

    estimator=make_shared<Estimator>();
    string sConfig_file = sConfig_file_ + "euroc_config.yaml";

    cout << "1 System() sConfig_file: " << sConfig_file << endl;
    readParameters(sConfig_file);

    ofs_pose.open("./pose_output.txt",fstream::out);
    if(!ofs_pose.is_open())
    {
        cerr << "ofs_pose is not open" << endl;
    }
    trackerData[0].readIntrinsicParameter(sConfig_file);

    estimator->setParameter();

    pgbuilder=make_shared<PoseGraphBuilder>(estimator);

    cout << "2 System() end" << endl;
}

System::~System()
{
    cout<<"system exist"<<endl;
    bStart_backend = false;
    
    m_buf.lock();
    while (!feature_buf.empty())
        feature_buf.pop();
    while (!imu_buf.empty())
        imu_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator->clearState();
    m_estimator.unlock();

    ofs_pose.close();
    cv::destroyAllWindows();

}

void System::PubImageData(double dStampSec, Mat &img)
{
    if (!init_feature)
    {
        cout << "1 PubImageData skip the first detected IDsfeatures, which doesn't contain optical flow speed" << endl;
        init_feature = 1;
        return;
    }

    if (first_image_flag)
    {
        cout << "2 PubImageData first_image_flag" << endl;
        first_image_flag = false;
        first_image_time = dStampSec;
        last_image_time = dStampSec;
        return;
    }
    // detect unstable camera stream
    if (dStampSec - last_image_time > 1.0 || dStampSec < last_image_time)
    {
        cerr << "3 PubImageData image discontinue! reset the IDsfeatures tracker!" << endl;
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        return;
    }
    last_image_time = dStampSec;
    // frequency control
    if (round(1.0 * pub_count / (dStampSec - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (dStampSec - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = dStampSec;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    TicToc t_r;
    trackerData[0].readImage(img, dStampSec);
    cvImgStampedPtr cvImg(new cvImgStamped(dStampSec,img));
    pgbuilder->GrabImg(cvImg);

    if (PUB_THIS_FRAME)
    {
        pub_count++;
        shared_ptr<IMG_MSG> feature_points(new IMG_MSG());
        feature_points->header = dStampSec;
        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    double x = un_pts[j].x;
                    double y = un_pts[j].y;
                    double z = 1;
                    feature_points->points.push_back(Vector3d(x, y, z));
                    feature_points->id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points->u_of_point.push_back(cur_pts[j].x);
                    feature_points->v_of_point.push_back(cur_pts[j].y);
                    feature_points->velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points->velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }

            if (!init_pub)
            {
                cout << "4 PubImage init_pub skip the first image!" << endl;
                init_pub = 1;
            }
            else
            {
                m_buf.lock();
                feature_buf.push(feature_points);
                m_buf.unlock();
                con.notify_one();
            }
        }
    }
}

vector<pair<vector<ImuConstPtr>, ImgConstPtr>> System::getMeasurements()
{
    vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
        {
            return measurements;
        }

        if (!(imu_buf.back()->header > feature_buf.front()->header + estimator->td))
        {
            cerr << "wait for imu, only should happen at the beginning sum_of_wait: " 
                << sum_of_wait << endl;
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header < feature_buf.front()->header + estimator->td))
        {
            cerr << "throw img, only should happen at the beginning" << endl;
            feature_buf.pop();
            continue;
        }
        ImgConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        vector<ImuConstPtr> IMUs;
        while (imu_buf.front()->header < img_msg->header + estimator->td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty()){
            cerr << "no imu between two image" << endl;
        }
        measurements.emplace_back(IMUs, img_msg);
        return measurements;
    }

}

void System::PubImuData(double dStampSec, const Eigen::Vector3d &vGyr, 
    const Eigen::Vector3d &vAcc)
{
    shared_ptr<IMU_MSG> imu_msg(new IMU_MSG());
	imu_msg->header = dStampSec;
	imu_msg->linear_acceleration = vAcc;
	imu_msg->angular_velocity = vGyr;

    if (dStampSec <= last_imu_t)
    {
        cerr << "imu message in disorder!" << endl;
        return;
    }
    last_imu_t = dStampSec;
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();
}

void System::ProcessBackEnd()
{
    auto time=std::chrono::system_clock::now();
    while (bStart_backend)
    {
        vector<pair<vector<ImuConstPtr>, ImgConstPtr>> measurements;
        
        unique_lock<mutex> lk(m_buf);
        con.wait_for(lk,std::chrono::seconds(5),[&] {
            if((measurements=getMeasurements()).size() != 0){
                return true;
            }
            else return false;
        });

        if( measurements.size() > 1){
        cout << "1 getMeasurements size: " << measurements.size() 
            << " imu sizes: " << measurements[0].first.size()
            << " feature_buf size: " <<  feature_buf.size()
            << " imu_buf size: " << imu_buf.size() << endl;
        }
        lk.unlock();
        m_estimator.lock();
        {
            std::unique_lock<std::mutex> lock(estimator->m_front_terminated);
            if(measurements.empty()){
                estimator->front_terminated=true;
                cout<<"ProcessBackEnd end"<<endl;
                return ;
            }
        }
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header;
                double img_t = img_msg->header + estimator->td;
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    assert(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x();
                    dy = imu_msg->linear_acceleration.y();
                    dz = imu_msg->linear_acceleration.z();
                    rx = imu_msg->angular_velocity.x();
                    ry = imu_msg->angular_velocity.y();
                    rz = imu_msg->angular_velocity.z();
                    estimator->processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else
                {
                    cout<<"why this happened?"<<endl;
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    assert(dt_1 >= 0);
                    assert(dt_2 >= 0);
                    assert(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x();
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y();
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z();
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x();
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y();
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z();
                    estimator->processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }

            // set relocalization frame
            {
                RelocinfoConstPtr relo_msg;
                pgbuilder->m_reloc_buf.lock();
                while (!pgbuilder->relo_buf.empty())
                {
                    relo_msg = pgbuilder->relo_buf.front();
                    pgbuilder->relo_buf.pop();
                }
                pgbuilder->m_reloc_buf.unlock();
                if (relo_msg!= nullptr)
                {
                    vector<Vector3d> match_points;
                    double frame_stamp = relo_msg->header;
                    for (unsigned int i = 0; i < relo_msg->uv_old_norm.size(); i++)
                    {
                        Vector3d u_v_id;
                        u_v_id.x() = relo_msg->uv_old_norm[i].x();
                        u_v_id.y() = relo_msg->uv_old_norm[i].y();
                        u_v_id.z() = relo_msg->matched_id[i];
                        match_points.push_back(u_v_id);
                    }
                    estimator->setReloFrame(frame_stamp, relo_msg->index, match_points, relo_msg->t_old, relo_msg->R_old);
                }
                estimator->m_loop_buf.lock();
                if(!estimator->loop_buf.empty()){
                    auto loop=estimator->loop_buf.front();
                    pgbuilder->GrabRelocRelativePose(loop.first,loop.second);
                    estimator->loop_buf.pop();
                }
                estimator->m_loop_buf.unlock();

            }

            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++) 
            {
                int v = img_msg->id_of_point[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x();
                double y = img_msg->points[i].y();
                double z = img_msg->points[i].z();
                double p_u = img_msg->u_of_point[i];
                double p_v = img_msg->v_of_point[i];
                double velocity_x = img_msg->velocity_x_of_point[i];
                double velocity_y = img_msg->velocity_y_of_point[i];
                assert(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            TicToc t_processImage;
            estimator->processImage(image, img_msg->header);
            
            if (estimator->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                if(estimator->marginalization_flag == Estimator::MARGIN_OLD){

//                    double dkeystamp=estimator->Headers[0];
//                    Eigen::Matrix3d keyR=estimator->Rs[0];
//                    Eigen::Vector3d keyT=estimator->Ps[0];
//                    PosestampedPtr pose(new PoseStamped(dkeystamp,keyR,keyT));
//                    pgbuilder->GrabKeyFramePose(pose);

                    estimator->m_pose_graph_buf.lock();
                    double dkeystamp;
                    if(!estimator->pose_graph_factors_buf.empty()){
                        auto kf=estimator->pose_graph_factors_buf.front();
                        pgbuilder->GrabKeyFrameFactor(kf);
                        dkeystamp=kf->ts;
                        estimator->pose_graph_factors_buf.pop();
                    }
                    estimator->m_pose_graph_buf.unlock();

                    Keyframe_pointsPtr kfp(new keyframe_points);
                    kfp->header=dkeystamp;
                    for (auto &idFeatures : estimator->f_manager.IDsfeatures)
                    {
                        int frame_size = idFeatures.idfeatures.size();
                        if(idFeatures.start_frame ==0 && idFeatures.start_frame + frame_size - 1 >= Vo_SIZE - 3 && idFeatures.solve_flag == 1)
                        {
                            int imu_i = idFeatures.start_frame;
                            Vector3d pts_i = idFeatures.idfeatures[0].point * idFeatures.estimated_depth;
                            Vector3d w_pts_i = estimator->Rs[imu_i] * (estimator->ric[0] * pts_i + estimator->tic[0])
                                               + estimator->Ps[imu_i];
                            Eigen::Vector3d p;
                            p.x() = w_pts_i(0);
                            p.y() = w_pts_i(1);
                            p.z() = w_pts_i(2);
                            kfp->points.push_back(p);

                            int imu_j = 0;
                            Eigen::Vector2d uv_p,norm_p;
                            uv_p.x()=idFeatures.idfeatures[imu_j].uv.x();
                            uv_p.y()=idFeatures.idfeatures[imu_j].uv.y();
                            norm_p.x()=idFeatures.idfeatures[imu_j].point.x();
                            norm_p.y()=idFeatures.idfeatures[imu_j].point.y();
                            kfp->norm_points.push_back(norm_p);
                            kfp->uv_points.push_back(uv_p);
                            kfp->vids.push_back(idFeatures.feature_id);

                        }

                    }
                    pgbuilder->GrabKeyframePoints(kfp);
                }

            }

            if (estimator->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            {
                Vector3d p_wi;
                Quaterniond q_wi;
                q_wi = Quaterniond(estimator->Rs[0]);
                p_wi = estimator->Ps[0];
                vPath_to_draw.push_back(p_wi);
                double dStamp = estimator->Headers[0];
                ofs_pose << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " "
                         << q_wi.w() << " " << q_wi.x() << " " << q_wi.y() << " " << q_wi.z() << endl;
            }
        }
        m_estimator.unlock();
    }
}

void System::Draw() 
{   
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_camp = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_camp = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_camp));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_camp.Activate(s_camp);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);
         
        // draw poses
        glColor3f(0, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        int nPath_size = vPath_to_draw.size();
        for(int i = 0; i < nPath_size-1; ++i)
        {        
            glVertex3f(vPath_to_draw[i].x(), vPath_to_draw[i].y(), vPath_to_draw[i].z());
            glVertex3f(vPath_to_draw[i+1].x(), vPath_to_draw[i+1].y(), vPath_to_draw[i+1].z());
        }
        glEnd();

        string marg=" NEW";
        if(estimator->marginalization_flag==Estimator::MARGIN_OLD){
            marg="OLD";
        }
        glColor3f(1.0, 0.0, 0.0);
        pangolin::GlFont::I()
                .Text("marg %s",marg.c_str()).Draw(5, 20);

        if(estimator->relocalization_info){
            glColor3f(1.0, 0.0, 0.0);
            pangolin::GlFont::I()
                    .Text("VIO prefrom relocation").Draw(5, 20);
        }

        // points
        if (estimator->solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            glPointSize(5);
            glBegin(GL_POINTS);
            for(int i = 0; i < ALL_BUF_SIZE;++i)
            {
                Vector3d p_wi = estimator->Ps[i];
                glColor3f(1, 0, 0);
                glVertex3d(p_wi[0],p_wi[1],p_wi[2]);
            }
            glEnd();
        }

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
