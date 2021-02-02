#include "pose_graph/pose_graph_builder.h"

void PoseGraphBuilder::new_sequence(){
    printf("new sequence\n");
    sequence++;
    printf("sequence cnt %d \n", sequence);
    if (sequence > 5)
    {
        cerr<<"only support 5 sequences since it's boring to copy code for more sequences.";
    }
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pgfactor_buf.empty())
        pgfactor_buf.pop();
    m_buf.unlock();
}

void PoseGraphBuilder::GrabImg(cvImgStampedPtr &imgstamped){
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    image_buf.push(imgstamped);
    m_buf.unlock();
    int stamp=imgstamped->header;
    // detect unstable camera stream
    if (last_image_time == -1)
        last_image_time = stamp;
    else if (stamp - last_image_time > 1.0 || stamp < last_image_time)
    {
        cerr<<"image discontinue! detect a new sequence!"<<endl;
        new_sequence();
    }
    last_image_time = stamp;
}

void PoseGraphBuilder::GrabKeyframePoints(Keyframe_pointsPtr &kps){
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    point_buf.push(kps);
    m_buf.unlock();
}

void PoseGraphBuilder::GrabKeyFrameFactor(CombinedFactors *factor){
    if(!LOOP_CLOSURE)
        return;
    m_buf.lock();
    pgfactor_buf.push(factor);
    m_buf.unlock();
}

void PoseGraphBuilder::process()
{

    if (!LOOP_CLOSURE)
        return;
    bool finish=false;
    auto tm_time=std::chrono::system_clock::now();
    auto empty_time=tm_time;
    accumFactor=new CombinedFactors();
    long pg_index=0;
    while (true)
    {
        if(estimator->solver_flag==Estimator::NON_LINEAR){

            {
                std::unique_lock<std::mutex> lock(estimator->m_front_terminated);
                if(estimator->front_terminated){
                    finish=true;
                }
            }
            cvImgStampedConstPtr image_msg= nullptr;
            CombinedFactors *factor_msg= nullptr;
            Keyframe_pointsConstPtr point_msg= nullptr;

            // find out the messages with same time stamp
            m_buf.lock();
            if(!image_buf.empty() && !point_buf.empty() && !pgfactor_buf.empty())
            {
                if(finish){
                    tm_time=std::chrono::system_clock::now();
                }

                if (image_buf.front()->header > pgfactor_buf.front()->ts)
                {
                    pgfactor_buf.pop();
                    printf("throw pose at beginning\n");
                }
                else if (image_buf.front()->header> point_buf.front()->header)
                {

                    point_buf.pop();
                    printf("throw point at beginning\n");
                }
                else if (image_buf.back()->header >= pgfactor_buf.front()->ts
                         && point_buf.back()->header >= pgfactor_buf.front()->ts)
                {
                    factor_msg=pgfactor_buf.front();
                    pgfactor_buf.pop();
                    while (!pgfactor_buf.empty()){
                        pgfactor_buf.pop();
                    }

                    while (image_buf.front()->header < factor_msg->ts)
                        image_buf.pop();
                    image_msg = image_buf.front();
                    image_buf.pop();

                    while (point_buf.front()->header < factor_msg->ts)
                        point_buf.pop();
                    point_msg = point_buf.front();
                    point_buf.pop();
                }
            }else{
                empty_time=std::chrono::system_clock::now();
            }
            m_buf.unlock();


            if(std::chrono::duration<double>(empty_time-tm_time).count()>5 && finish){
                m_terminate.lock();
                terminate=true;
                m_terminate.unlock();
                return;
            }


            if (factor_msg)
            {
                //printf(" pose time %f \n", pose_msg.first);
                //printf(" point time %f \n", point_msg.first);
                //printf(" image time %f \n", image_msg.first);
                // skip fisrt few
//                if (skip_first_cnt < SKIP_FIRST_CNT)
//                {
//                    skip_first_cnt++;
//                    continue;
//                }
//
//                if (skip_cnt < SKIP_CNT)
//                {
//                    skip_cnt++;
//                    continue;
//                }
//                else
//                {
//                    skip_cnt = 0;
//                }

                // build keyframe
                currentFactor=factor_msg;
                *accumFactor=(*accumFactor)+(*currentFactor);
                if(accumFactor->distance>0.1){
                    //debug
//                    cout<<"test--------------------"<<endl;
//                    cout<<"PoseGraph index "<<accumFactor->pg_index<<endl;
//                    cout<<"GV info ::"<<endl<<accumFactor->rollPitchFactor->sqrt_info<<endl;

                    Eigen::Matrix3d R=accumFactor->Ri;
                    Eigen::Vector3d T=accumFactor->ti;

                    vector<cv::Point3f> point_3d;
                    vector<cv::Point2f> point_2d_uv;
                    vector<cv::Point2f> point_2d_normal;
                    vector<double> point_id;

                    for (unsigned int i = 0; i < point_msg->points.size(); i++)
                    {
                        cv::Point3f p_3d;
                        p_3d.x = point_msg->points[i].x();
                        p_3d.y = point_msg->points[i].y();
                        p_3d.z = point_msg->points[i].z();
                        point_3d.push_back(p_3d);

                        cv::Point2f p_2d_uv, p_2d_normal;
                        double p_id;
                        p_2d_normal.x = point_msg->norm_points[i].x();
                        p_2d_normal.y = point_msg->norm_points[i].y();
                        p_2d_uv.x = point_msg->uv_points[i].x();
                        p_2d_uv.y = point_msg->uv_points[i].y();
                        p_id = point_msg->vids[i];
                        point_2d_normal.push_back(p_2d_normal);
                        point_2d_uv.push_back(p_2d_uv);
                        point_id.push_back(p_id);

                        //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
                    }

                    if(last_kf){
                        Matrix3d Rj=last_kf->origin_vio_R*last_kf->keyfactor->relativePoseFactor->delta_R;
                        Vector3d tj=last_kf->origin_vio_T+last_kf->origin_vio_R*last_kf->keyfactor->relativePoseFactor->delta_t;
                        last_kf->keyfactor->relativePoseFactor->update(last_kf->origin_vio_T, last_kf->origin_vio_R,
                                                                       tj, Rj,
                                                                       accumFactor->ti, accumFactor->Ri);
                    }
                    KeyFrame* keyframe = new KeyFrame(accumFactor,image_msg->img,point_3d,point_2d_uv,point_2d_normal,point_id,sequence);
                    last_kf=keyframe;
                    m_process.lock();
                    start_flag = 1;
                    posegraph.addKeyFrame(keyframe, true);
                    if(keyframe->relc_info!= nullptr){
                        m_reloc_buf.lock();
                        relo_buf.push(keyframe->relc_info);
                        keyframe->relc_info.reset();
                        m_reloc_buf.unlock();
                    }
                    m_process.unlock();

                    //delete accumFactor;
                    accumFactor=new CombinedFactors(++pg_index);
                }

            }

        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraphBuilder::Draw()
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("PoseGraph Viewer", 1920, 1080);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1920, 1080, 500, 500, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(-5, 0, 15, 7, 0, 0, 1.0, 0.0, 0.0)
    );

    d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1920.0f / 1080.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false && !terminate) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.75f, 0.75f, 0.75f, 0.75f);
        glColor3f(0, 0, 1);
        pangolin::glDrawAxis(3);

        posegraph.m_keyframelist.lock();
        list<KeyFrame*>::iterator it;

        Vector3d P=Eigen::Vector3d::Zero(),lastP=Eigen::Vector3d::Zero();
        Matrix3d R=Eigen::Matrix3d::Identity();
        for (it = posegraph.keyframelist.begin(); it != posegraph.keyframelist.end(); it++)
        {

           (*it)->getPose(P, R);
            Quaterniond Q;
            Q = R;
            // draw poses
            glColor3f(0, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(lastP.x(),lastP.y(),lastP.z());
            glVertex3f(P.x(),P.y(),P.z());
            glEnd();

            if((*it)->cov_computed){
                //cerr<<"cov_computed: "<<(*it)->index<<endl;
                auto cov=(*it)->cov;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> SAE(cov);
                Vector3d xyz=SAE.eigenvalues().tail<3>();
                MatrixXd ev=SAE.eigenvectors().real().rightCols(3);

                double thex=atan(ev(1,0)/ev(0,0));
                Eigen::Matrix3d R1=Eigen::AngleAxisd(thex,Eigen::Vector3d::UnitZ()).toRotationMatrix();
                vector<Vector3d> ells;
                float res=5*M_PI/180.0;
                for (int j = 0; j <180 ; ++j) {
                    float x=10.0*sqrt(7.9*xyz.x())*cos(2.0*float(j)*res);
                    float y=10.0*sqrt(7.9*xyz.y())*sin(2.0*float(j)*res);
                    Vector3d ell=R1*Vector3d(x,y,0)+Vector3d(P.x(),P.y(),P.z());
                    ells.push_back(ell);
                }

                for (int i = 0; i <ells.size()-1 ; ++i) {
                    int j=i+1;
                    glColor3f(0, 1, 0);
                    glLineWidth(2);
                    glBegin(GL_LINES);
                    glVertex3f(ells[i].x(),ells[i].y(),ells[i].z());
                    glVertex3f(ells[j].x(),ells[j].y(),ells[j].z());
                    glEnd();
                }


            }


            Vector3d p_wi;
            Quaterniond q_wi;
            q_wi =R;
            p_wi =P;

            if (SHOW_L_EDGE)
            {
                if ((*it)->has_loop && (*it)->sequence == posegraph.sequence_cnt)
                {

                    KeyFrame* connected_KF = posegraph.getKeyFrame((*it)->loop_index);
                    Vector3d connected_P;
                    Matrix3d connected_R;
                    connected_KF->getPose(connected_P, connected_R);
                    //(*it)->getVioPose(P, R);
                    (*it)->getPose(P, R);
                    if((*it)->sequence > 0)
                    {
                        glColor3f(1, 0, 0);
                        glLineWidth(2);
                        glBegin(GL_LINES);
                        glVertex3f(P.x(),P.y(),P.z());
                        glVertex3f(connected_P.x(),connected_P.y(),connected_P.z());
                        glEnd();
                    }
                }
            }
            lastP=P;

        }
        posegraph.m_keyframelist.unlock();

        pangolin::FinishFrame();
        usleep(5);   // sleep 5 ms
    }
}