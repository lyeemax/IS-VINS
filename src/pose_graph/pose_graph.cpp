#include "pose_graph/pose_graph.h"

PoseGraph::PoseGraph()
{
	t_optimization = std::thread(&PoseGraph::optimizeCS, this);
	t_optimization.detach();
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    global_index = 0;
    sequence_cnt = 0;
    sequence_loop.push_back(0);
    base_sequence = 1;

}

void PoseGraph::loadVocabulary(std::string voc_path)
{
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    //shift to base frame
    Vector3d vio_P_cur;
    Matrix3d vio_R_cur;
    if (sequence_cnt != cur_kf->sequence)
    {
        sequence_cnt++;
        sequence_loop.push_back(0);
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        m_drift.lock();
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();
    }
    
    cur_kf->getVioPose(vio_P_cur, vio_R_cur);
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
    vio_R_cur = w_r_vio *  vio_R_cur;//Rww'*Rw'i
    cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
	int loop_index = -1;
    if (flag_detect_loop)
    {
        TicToc tmp_t;
        loop_index = detectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }
	if (loop_index != -1)
	{
        //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFrame* old_kf = getKeyFrame(loop_index);

        if (cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

//            Vector3d w_P_old, w_P_cur, vio_P_cur;
//            Matrix3d w_R_old, w_R_cur, vio_R_cur;
//            old_kf->getVioPose(w_P_old, w_R_old);
//            cur_kf->getVioPose(vio_P_cur, vio_R_cur);
//
//            Vector3d relative_t;
//            Quaterniond relative_q;
//            relative_t = cur_kf->getLoopRelativeT();//tlc
//            relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();//Rlc
//            w_P_cur = w_R_old * relative_t + w_P_old;
//            w_R_cur = w_R_old * relative_q;
//            Matrix3d shift_r;//Rww'
//            Vector3d shift_t;//tww'
//            shift_r=w_R_cur*vio_R_cur.transpose();
//            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
//            // shift vio pose of whole sequence to the world frame
//            if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
//            {
//                w_r_vio = shift_r;
//                w_t_vio = shift_t;
//                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
//                vio_R_cur = w_r_vio *  vio_R_cur;
//                cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
//                list<KeyFrame*>::iterator it = keyframelist.begin();
//                for (; it != keyframelist.end(); it++)
//                {
//                    if((*it)->index>loop_index)
//                    {
//                        Vector3d vio_P_cur;
//                        Matrix3d vio_R_cur;
//                        (*it)->getVioPose(vio_P_cur, vio_R_cur);
//                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
//                        vio_R_cur = w_r_vio *  vio_R_cur;
//                        (*it)->updateVioPose(vio_P_cur, vio_R_cur);
//                    }
//                }
//                sequence_loop[cur_kf->sequence] = 1;
//            }
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index);
            m_optimize_buf.unlock();
        }
	}
	m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getVioPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->updatePose(P, R);
	keyframelist.push_back(cur_kf);
	m_keyframelist.unlock();
}

KeyFrame* PoseGraph::getKeyFrame(int index)
{
//    unique_lock<mutex> lock(m_keyframelist);
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }
    TicToc tmp_t;
    //first query; then add this frame into database!
    QueryResults ret;
    TicToc t_query;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index-50);
    //printf("query time: %f", t_query.toc());
    //cout << "Searching for Image " << frame_index << ". " << ret << endl;

    TicToc t_add;
    db.add(keyframe->brief_descriptors);
    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result 
    if (DEBUG_IMAGE)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(loop_result, tmp_image, loop_result);
        }
    }
    // a good match with its neighbour
    if (ret.size() >= 1 &&ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true;
                int tmp_index = ret[i].Id;
                if (DEBUG_IMAGE)
                {
                    auto it = image_pool.find(tmp_index);
                    cv::Mat tmp_image = (it->second).clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                }
            }

        }
    if (DEBUG_IMAGE)
    {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }
    if (find_loop && frame_index > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;

}

void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[keyframe->index] = compressed_image;
    }

    db.add(keyframe->brief_descriptors);
}
void PoseGraph::optimizeCS() {
    while(true)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while(!optimize_buf.empty())
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();

        if (cur_index != -1)
        {
            //printf("optimize pose graph \n");
            //cout<<"optimize between "<<first_looped_index<<" and "<<cur_index<<endl;
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame* cur_kf = getKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double pose_array[max_length][7];

            ceres::Covariance::Options coptions;
            ceres::Covariance covariance(coptions);
            ceres::Problem::Options problemOptions;
            problemOptions.cost_function_ownership=ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
            ceres::Problem problem(problemOptions);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 10;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);

            list<KeyFrame*>::iterator it;

            int param_index = 0;
            vector<pair<const double*, const double*> > covariance_blocks;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;

                (*it)->local_index = param_index;
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->getVioPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                tmp_q.normalize();
                pose_array[param_index][0] = tmp_t(0);
                pose_array[param_index][1] = tmp_t(1);
                pose_array[param_index][2] = tmp_t(2);
                pose_array[param_index][3] =tmp_q.x();
                pose_array[param_index][4] =tmp_q.y();
                pose_array[param_index][5] =tmp_q.z();
                pose_array[param_index][6] =tmp_q.w();
                ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                problem.AddParameterBlock(pose_array[param_index], 7, local_parameterization);
                //covariance_blocks.push_back(make_pair(pose_array[param_index], pose_array[param_index]));
                if ( (*it)->index== first_looped_index ||(*it)->sequence == 0)
                {
                    problem.SetParameterBlockConstant(pose_array[param_index]);
                }
                if ((*it)->index == cur_index)
                    break;
                param_index++;
            }

            for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
                if ((*it)->index < first_looped_index)
                    continue;
                if ((*it)->index== cur_index)
                    break;
                int li=(*it)->local_index;
                int lj=li+1;


                //add edge

                if((*it)->keyfactor->rollPitchFactor){
                    problem.AddResidualBlock((*it)->keyfactor->rollPitchFactor, NULL, pose_array[li]);
                }
                if(lj <= param_index){

                    problem.AddResidualBlock((*it)->keyfactor->relativePoseFactor, NULL, pose_array[li], pose_array[lj]);
                }

                //add loop edge

                if((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                    RelativePoseFactor* relocfactor=new RelativePoseFactor((*it)->getLoopRelativeT(),(*it)->getLoopRelativeQ().toRotationMatrix());
                    relocfactor->sqrt_info=1e5*Eigen::Matrix<double,6,6>::Identity();
                    problem.AddResidualBlock(relocfactor, loss_function, pose_array[connected_index], pose_array[li]);
                }


            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << "\n";
//            CHECK(covariance.Compute(covariance_blocks, &problem));
//            double covariance_pose[6 * 6];
//            covariance.GetCovarianceBlock(pose_array[cur_index-1], pose_array[cur_index-1], covariance_pose);
//            std::cerr<<Eigen::Map<Eigen::Matrix<double,6,6>>(covariance_pose)<<std::endl;

            //printf("pose optimization time: %f \n", tmp_t.toc());
            /*
            for (int j = 0 ; j < param_index; j++)
            {
                printf("optimize param_index: %d p: %f, %f, %f\n", j, pose_array[j][0], pose_array[j][1], pose_array[j][2] );
            }
            */
            m_keyframelist.lock();
            param_index = 0;
            KeyFrame* last= nullptr;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;


                Quaterniond tmp_q(pose_array[param_index][6], pose_array[param_index][3], pose_array[param_index][4], pose_array[param_index][5]);
                Vector3d tmp_t = Vector3d(pose_array[param_index][0], pose_array[param_index][1], pose_array[param_index][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)-> updatePose(tmp_t, tmp_r);
                if(last){
                    last->keyfactor->relativePoseFactor->update(last->T_w_i,last->R_w_i,(*it)->T_w_i,(*it)->R_w_i,
                                                                 pose_array[param_index-1],pose_array[param_index]);
                }

                last=*it;
                if ((*it)->index == cur_index)
                    break;
                param_index++;
            }

            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->getPose(cur_t, cur_r);
            cur_kf->getVioPose(vio_t, vio_r);
            m_drift.lock();
            yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
            r_drift = cur_r*vio_r.transpose();
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            cout << "t_drift " << t_drift.transpose() << endl;
            cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;

            it++;
            for (; it != keyframelist.end(); it++)
            {
                Vector3d P;
                Matrix3d R;
                (*it)->getVioPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->updatePose(P, R);
            }

            m_keyframelist.unlock();
        }

        std :: ofstream ofs("./loop_pose_output.txt",std :: ios :: out | std :: ios :: trunc);
        for (auto it = keyframelist.begin(); it != keyframelist.end(); it++) {
            double dStamp =(*it)->time_stamp;
            Vector3d p_wi;
            Matrix3d R_wi;
            (*it)->getPose(p_wi, R_wi);
            Quaterniond q_wi(R_wi);
            ofs << fixed << dStamp << " " << p_wi(0) << " " << p_wi(1) << " " << p_wi(2) << " "
                << q_wi.w() << " " << q_wi.x() << " " << q_wi.y() << " " << q_wi.z() << endl;

        }
        ofs.close();

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}