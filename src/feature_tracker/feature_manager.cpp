#include "feature_tracker/feature_manager.h"

int IDFeatures::endFrame()
{
    return start_frame + idfeatures.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    IDsfeatures.clear();
}
bool FeatureManager::goodFeature(IDFeatures &idfs){//#TODO:how to define a good criterion
    bool optimizable=idfs.used_num >= 2&& idfs.start_frame < Vo_SIZE;
    //bool significant=idfs.used_num >= 2&&idfs.significant;
    return optimizable;
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &idfs : IDsfeatures)
    {

        idfs.used_num = idfs.idfeatures.size();

        if (goodFeature(idfs))
        {
            cnt++;
        }
    }
    return cnt;
}

//map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image
//ID,points
//frame_count:出现在哪一个frame上

//对出现在最新帧上的所有特征点检查视差（和窗口内所有特征对比视差）
bool FeatureManager::addFeatureAndCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    //取出图像上的所有特征点，加入特征管理器中
    for (auto &id_pts : image)
    {
        Feature feature(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        auto it = find_if(IDsfeatures.begin(), IDsfeatures.end(), [feature_id](const IDFeatures &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == IDsfeatures.end())
        {
            IDsfeatures.push_back(IDFeatures(feature_id, frame_count));//start_frame 是 frame_count
            IDsfeatures.back().idfeatures.push_back(feature);
        }
        else if (it->feature_id == feature_id)
        {
            it->idfeatures.push_back(feature);
            last_track_num++;
        }
    }
//如果系统刚开始运行或者跟踪质量比较差了
    if (frame_count < 2 || last_track_num < 20)
        return true;
    for (auto &idFeatures : IDsfeatures)
    {
        if (idFeatures.start_frame <= frame_count - 2 &&
            idFeatures.start_frame + int(idFeatures.idfeatures.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(idFeatures, frame_count);
            parallax_num++;
        }
    }
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    //ROS_DEBUG("debug show");
    for (auto &it : IDsfeatures)
    {
        assert(it.idfeatures.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        //ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.idfeatures)
        {
            //ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}
//返回出现在frame_count_l和frame_count_r上的所有特征点集合
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &idFeatures : IDsfeatures)
    {
        if (idFeatures.start_frame <= frame_count_l && idFeatures.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - idFeatures.start_frame;
            int idx_r = frame_count_r - idFeatures.start_frame;

            a = idFeatures.idfeatures[idx_l].point;

            b = idFeatures.idfeatures[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &idFeatures : IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!goodFeature(idFeatures))
            continue;

        idFeatures.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("IDsfeatures id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (idFeatures.estimated_depth < 0 ||idFeatures.estimated_depth>10)
        {
            idFeatures.solve_flag = 2;
        }
        else
            idFeatures.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = IDsfeatures.begin(), it_next = IDsfeatures.begin();
         it != IDsfeatures.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2 ||it->is_outlier)
            IDsfeatures.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &idFeatures : IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!goodFeature(idFeatures))
            continue;
        idFeatures.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &idFeatures : IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!goodFeature(idFeatures))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / idFeatures.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &idFeatures : IDsfeatures)
    {
        idFeatures.used_num = idFeatures.idfeatures.size();
        if (!goodFeature(idFeatures))
            continue;

        if (idFeatures.estimated_depth > 0 ||idFeatures.is_outlier)
            continue;
        int imu_i = idFeatures.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * idFeatures.idfeatures.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &feature : idFeatures.idfeatures)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = feature.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        idFeatures.estimated_depth = svd_method;
//        Vector3d pi=svd_method*idFeatures.idfeatures[0].point;
//        if(pi.norm()>8)
//            idFeatures.is_outlier=true;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (idFeatures.estimated_depth < 0.1 || idFeatures.estimated_depth >8.0)
        {
            idFeatures.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    int i = -1;
    for (auto it = IDsfeatures.begin(), it_next = IDsfeatures.begin();
         it != IDsfeatures.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            IDsfeatures.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = IDsfeatures.begin(), it_next = IDsfeatures.begin();
         it != IDsfeatures.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else//滑动窗口的第一帧
        {
            Eigen::Vector3d uv_i = it->idfeatures[0].point;
            it->idfeatures.erase(it->idfeatures.begin());//marg的时候丢掉该帧的观测，就是这么来的，
            if (it->idfeatures.size() < 2)
            {
                IDsfeatures.erase(it);
                continue;
            }
            else//留下来的点host帧是下一个了,所以在host的深度要调整
            {//这个时候start_frame还是0，只是薇姿和深度要调整
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost IDsfeatures after marginalize
        /*
        if (it->endFrame() < Vo_SIZE - 1)
        {
            IDsfeatures.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = IDsfeatures.begin(), it_next = IDsfeatures.begin();
         it != IDsfeatures.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->idfeatures.erase(it->idfeatures.begin());
            if (it->idfeatures.size() == 0)
                IDsfeatures.erase(it);
        }
    }
}


void FeatureManager::removeFront(int frame_count)
{
    for (auto it = IDsfeatures.begin(), it_next = IDsfeatures.begin(); it != IDsfeatures.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)//why?，这样理解：删除的是倒数第二帧，倒数第一帧上的点要往前摞
        {
            it->start_frame--;
        }
        else
        {
            int j = ALL_BUF_SIZE-1 - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->idfeatures.erase(it->idfeatures.begin() + j);
            if (it->idfeatures.size() == 0)
                IDsfeatures.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const IDFeatures &idFeatures, int frame_count)
{
    //check the first frame in Vio_SZIE is keyframe or not
    //parallax between  last frame in WINDOWS and first frame in Vio_SZIE
    const Feature &feature_l2 = idFeatures.idfeatures[frame_count - 2 -idFeatures.start_frame];
    const Feature &feature_l1 = idFeatures.idfeatures[frame_count - 1 - idFeatures.start_frame];

    double ans = 0;
    Vector3d p_l1 = feature_l1.point;

    double u_1 = p_l1(0);
    double v_1 = p_l1(1);

    Vector3d p_l2 = feature_l2.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_l2;
    double dep_i = p_l2(2);
    double u_i = p_l2(0) / dep_i;
    double v_i = p_l2(1) / dep_i;
    double du = u_i - u_1, dv = v_i - v_1;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_1, dv_comp = v_i_comp - v_1;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}