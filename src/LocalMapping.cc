/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

/**
 * @brief 本地Map线程的主循环
 * 
 * 在System::System()中创建本地Map线程并启动，然后进入这个主循环，处理Tracking中新添加的关键帧，
 * 为关键帧构造Map点，添加到本地Map，所有关键帧添加完成后进行本地Map的BA优化，更新关键帧位姿，更新Map点坐标
 * 
 * 1. 检查是否有新添加的关键帧
 *   a. 取出一帧新添加的关键帧，作为当前关键帧，进行处理，计算BOW，将关键帧插入Map
 *   b. 对当前关键帧的Map点进行检查，剔除不合格Map点
 *   c. 在当前关键帧和相邻帧之间搜索同时满足ORB匹配和对极约束的特征点对儿，构造更多的Map点
 * 2. 对当前关键帧和一级、二级相邻关键帧之间重复的Map点进行融合
 * 3. 进行本地Map的BA优化，对本地关键帧的位姿和本地Map点的坐标进行优化
 * 4. 剔除冗余关键帧
 * 5. 将当前关键帧添加到LoopCloser中，【注意】这是LoopCloser的唯一输入，
 *    添加到LocalMapper的所有关键帧都会被添加到LoopCloser，并同步添加到关键帧数据库
 */
void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        /**
         * 告诉Tracking：LocalMapping处于繁忙状态
         * 在LocalMapping没有处理完新的关键帧之前，防止Tracking创建新的关键帧。
         */
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        /* 确认有新的关键帧可以处理 */
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            /*取出一个关键帧，用mpCurrentKeyFrame指向，计算关键帧的词袋，将关键帧插入Map */
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            /* 检查Track中为立体视觉新添加的Map点，保留那些被很多帧看到的点，那些较少被看到的点被标识为坏点 */
            // Check recent MapPoints
            MapPointCulling();

            /* 在当前帧和相邻帧之间搜索同时满足ORB匹配和对极约束的特征点对儿，构造更多的Map点 */
            // Triangulate new MapPoints
            CreateNewMapPoints();

            /* 如果已经处理完队列中的最后一个关键帧 */
            if(!CheckNewKeyFrames())
            {
                /* 融合当前关键帧和一级、二级相邻关键帧之间重复的Map点 */
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            /* 如果已经处理完最后一个关键帧，并且没有人请求停止LocalMapping，进行本地Map的BA优化 */
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                /* 进行本地Map的BA，对本地关键帧的位姿和本地Map点的坐标进行优化 */
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                /* 剔除冗余关键帧，如果该关键帧的90%的Map点都能其他关键帧所看到，则认为冗余帧并剔除 */
                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            /* 将当前关键帧添加到回环检测中，【注意】这是LoopCloser的唯一输入 */
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        /* 如果有复位请求则删除所有新添加的关键帧和最近新添加的Map点 */
        ResetIfRequested();

        /* 解锁，Tracking可以添加新的关键帧了 */
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

/**
 * @brief 插入关键帧到本地Map中，设置中断BA的指示
 * 
 * @param pKF 待插入的关键帧
 */
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

/**
 * @brief 检查是否有新的关键帧
 * 
 */
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理新的关键帧
 * 
 * 1. 从队列中取得一帧关键帧，用mpCurrentKeyFrame指向
 * 2. 计算关键帧的词袋
 * 3. 取得关键帧的所有Map点并遍历：
 *   a. 如果当前关键帧不在Map点的观测中，则添加到观测中，更新Map点的平均观测方向和观测距离范围，更新该点的独特描述符
 *   b. 如果当前关键帧在Map点的观测中，则将点添加到mlpRecentAddedMapPoints中，等待检测
 * 4. 更新关键帧之间的连接关系
 * 5. 将关键帧插入地图中
 */
void LocalMapping::ProcessNewKeyFrame()
{
    /* 从队列中取得一帧新的关键帧 */
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    /* 计算该关键帧的词袋 */
    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    /* 取得该关键帧的所有Map点 */
    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    /* 遍历每一个Map点 */
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                /* 如果当前关键帧不在Map点的观测中，则添加 */
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    /* 添加该帧到Map点的观测中 */
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    /* 更新该点的平均观测方向和观测距离范围 */
                    pMP->UpdateNormalAndDepth();
                    /* 更新该点的独特描述符 */
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    /* 将双目或RGBD跟踪过程中新插入的Map点放入mlpRecentAddedMapPoints，等待检查，
                     * 这些点都会经过MapPointCulling函数的检验 */
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    /* 更新关键帧之间的连接关系 */
    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    /* 将关键帧插入地图中 */
    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 剔除ProcessNewKeyFrame添加的不合格Map点
 * 
 * 对新添加的Map点进行筛选，要求有尽可能多的图像帧真正看到该点，如果看到该点的图像帧偏少，
 * 则将该点标识为坏点。
 */
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            /* 如果实际看到该Map点的图像帧数和预计看到该Map点的帧数之比小于25%，剔除 */
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            /* 如果从创建该Map点开始已经有≥2个关键帧，但是观测到该点的关键帧却不超过2个（单目）或3个（非单目），剔除 */
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            /* 如果从创建该Map点开始已经过了不少于3个关键帧而没有被剔除，则认为是高质量的点
             * 因此没有将该点设置为坏点，但是从队列中删除，后续不再继续检测。 */
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

/**
 * @brief 创建新的Map点
 * 
 * 1. 取得与当前关键帧具有最佳共视效果的若干相邻关键帧
 * 2. 遍历每一个相邻关键帧
 *   a. 检查基线长度，基线长度不能太短
 *   b. 在当前帧和相邻帧之间搜索既满足ORB匹配且满足对极约束的特征点对儿
 *   c. 计算匹配特征点对儿的视差角
 *   d. 视差角小的时候用三角法恢复3D点，视差角大时用双目恢复3D点
 *   e. 检查生成的3D点是否在相机前方
 *   f. 检查3D点在当前帧和相邻帧下的重投影偏差
 *   g. 检查尺度一致性，即3D点到两个关键帧的距离基本相当
 *   h. 构造Map点，添加属性，添加到Map
 *   i. 放入待检测队列，后续用MapPointCulling进行检验
 */
void LocalMapping::CreateNewMapPoints()
{
    /* 取得与当前关键帧具有最佳共视效果的若干相邻关键帧 */
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    /* 取得从世界坐标系到当前关键帧相机坐标系的变换矩阵，以及当前关键帧的相机光心在世界坐标系下的坐标 */
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    /* 取得当前关键帧的相机内参 */
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    /* 遍历所有的相邻关键帧 */
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        /* 如果有新的关键帧则不创建新的Map点 */
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        /* 取得当前帧到相邻帧的相机位移长度，即基线长度 */
        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        /* 基线太短则不生成新的Map点 */
        if(!mbMonocular)
        {
            /* 对于非单目，关键帧的间距太小时不生成新的Map点 */
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            /* 对于单目，如果基线长度小于该关键帧景深的1%时不生成新的Map点 */
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        /* 计算关键帧之间的基本矩阵 */
        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        /* 搜索既满足ORB匹配且满足对极约束的特征点对儿 */
        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        /* 取得世界坐标系到相邻关键帧相机坐标系的转换矩阵 */
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        /* 取得相邻关键帧的相机内参 */
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            /* 将特征点kp1和kp2逆投影到各自的相机坐标系，得到各自相机坐标系下两个点xn1和xn2 */
            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            /* 将xn1和xn2转到世界坐标系，求从原点出发到这两个点的向量的夹角的Cos值 */
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            /* 对于双目，利用双目得到视差角 */
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            /**
             * cosParallaxRays<0.9998               表示特征点向量夹角大于1.146°
             * cosParallaxRays>0                    表示特征点向量夹角小于90°
             * cosParallaxRays<cosParallaxStereo    表示视差角很小
             * 视差角很小时用三角法恢复3D点，视差角大时用双目恢复3D点
             */
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                /* 利用来自两个图像的对应特征点以及各自的投影矩阵还原出空间点X的三维坐标
                 * 参见Initializer::Triangulate() */
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                /* 用双目恢复3D点 */
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                /* 用双目恢复3D点 */
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            /* 检查生成的3D点是否在相机前方 */
            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            /* 检查3D点在当前关键帧下的重投影偏差 */
            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            /* 检查3D点在相邻关键帧下的重投影偏差 */
            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            /* 检查尺度一致性，即3D点到两个关键帧的距离基本相当 */
            //Check scale consistency

            /* 3D点到当前关键帧的距离 */
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            /* 3D点到相邻关键帧的距离 */
            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            /* 两个距离的比值 */
            const float ratioDist = dist2/dist1;
            /* 两个特征点的金字塔尺度因子比值 */
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /* 尺度变化是连续的 */
            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            /* 三角化成功，构造Map点 */
            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            /* 为Map点添加属性 */
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            /* 添加到地图 */
            mpMap->AddMapPoint(pMP);

            /* 放入待检测队列，后续用MapPointCulling进行检验 */
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

/**
 * @brief 融合当前关键帧和一级、二级相邻关键帧之间重复的Map点
 * 
 * 1. 取得当前关键帧的20个一级相邻帧，以及每个相邻帧的5个二级相邻帧，总共100个相邻帧
 * 2. 为当前帧的每个Map点在一级和二级相邻帧中寻找匹配的特征点，如果匹配特征点已有Map点就合并，没有就添加
 * 3. 为一级和二级相邻帧的每个Map点在当前帧中寻找匹配的特征点，如果匹配特征点已有Map点就合并，没有就添加
 * 4. 更新当前关键帧的独特描述符、平均观测方向、观测范围以及连接
 */
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    
    /* 取得与当前帧具有最佳共视效果的20个（对于非单目是10个）一级相邻帧，以及5个二级相邻帧，总共是100个相邻帧，保存在vpTargetKFs中 */
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; 
        
        // mnFuseTargetForKF 唯一目的是防止重复添加相同的帧，别无他用

        /* 对每个相邻帧，再取与其具有最佳共视效果的5个相邻帧 */
        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    /**
     * 将当前帧的Map点与一级和二级相邻帧的Map点进行融合。
     * 为当前帧的每个Map点在一级和二级相邻帧中寻找匹配的特征点，
     * 找到后，如果该特征点已有Map点，则进行融合，如果没有，则将当前帧的Map点添加为该特征点的Map点。*/
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    /**
     * 将所有一级和二级相邻帧的Map点找出来，与当前帧的Map点进行融合
     * 方法与上面的融合完全一致，只不过方向相反 */
    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    /* 将所有一级和二级相邻帧的Map点找出来 */
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;  //mnFuseCandidateForKF变量的唯一作用是在这里防止重复添加Map点，别的地方没有使用
            vpFuseCandidates.push_back(pMP);
        }
    }

    /* 将相邻帧的Map点与当前帧的Map点进行融合 */
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    /* 更新当前帧的Map点，主要是独特描述符、平均观测方向以及观测范围 */
    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    /* 更新关键帧之间的连接，即更新与当前帧具有共视关系的关键帧 */
    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

/**
 * @brief 计算两个关键帧之间的基本矩阵
 * 
 * @param pKF1 关键帧1
 * @param pKF2 关键帧2
 * @return cv::Mat 基本矩阵
 */
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

/**
 * @brief 剔除冗余关键帧
 * 
 * 如果90%的Map点都能被至少3个其他关键帧看到，则该帧被认为是冗余帧
 * 
 */
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

/**
 * @brief 计算向量v的反对称矩阵
 * 
 * @param v 长度为3的向量
 * @return cv::Mat 反对称矩阵
 */
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
