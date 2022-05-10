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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

/**
 * @brief 回环闭合的主线程
 * 
 * 1. 检查是否有新的关键帧到达
 * 2. 进行回环检测
 * 3. 计算近似的变换矩阵
 * 4. 进行回环纠正
 */
void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            /* 在关键帧数据库中检测回环关键帧，并确认是否连续三个周期都检测到 */
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())
            {
                /*计算出当前帧与回环候选帧的空间变换Rt并搜索更多的匹配特征点*/
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   /* 进行闭环融合以及位姿图优化 */
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

/**
 * @brief 插入一个关键帧到LoopCloser中
 * 
 * @param pKF 关键帧
 */
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

/**
 * @brief 检查是否有新的关键帧到达
 * 
 * @return true 有新的关键帧到达
 * @return false 没有新的关键帧到达
 */
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

/**
 * @brief 在关键帧数据库中检测回环关键帧，并确认是否连续三个周期都检测到
 * 
 * 不论是否成功，都将当前关键帧添加到数据库，用于后续的回环检测和重定位，总共有【3】处
 * 
 * 1. 取得新添加的一个关键帧，设置为LoopCloser的当前关键帧
 * 2. 找出当前关键帧与其相邻帧的最低相似度分值，作为判断是否是回环的最低参考阈值
 * 3. 在关键帧数据库中寻找相似度分值比相邻帧相似度分值更高的关键帧，作为候选关键帧
 * 4. 如果候选关键帧所在的群组连续三个关键帧周期都被列为回环候选者，那么候选关键帧作为最终结果输出
 * 
 * 探测回环的关键点有两个：
 * 1. 在关键帧数据库中找到相似度足够高的，且不相邻的关键帧，作为候选帧
 * 2. 连续三个关键帧周期该候选帧所在的组都被列为回环候选者，即连续三帧都检测到同一个回环
 * 
 * @return true 有回环
 * @return false 没有回环
 */
bool LoopClosing::DetectLoop()
{
    /* 取得新添加的一个关键帧，设置LoopCloser的当前关键帧mpCurrentKF */
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    /* 如果总的关键帧不足10帧，或者距离上次闭环不足10帧，则不进行回环检测
     * 但是要将当前关键帧添加到数据库，以便后续的回环检测 */
    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF); //【1】添加当前关键帧到数据库，用于回环检测和重定位
        mpCurrentKF->SetErase();
        return false;
    }

    /* 遍历当前关键帧的所有相邻帧，计算每个相邻帧与当前帧的BoW相似度分数，并记录相似度最低的分值minScore */
    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    /* 在关键帧数据库中寻找相似度分值比相邻帧相似度分值minScore更高的关键帧，作为候选关键帧 */
    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    /* 如果在关键帧数据库中没有找分值更高的关键帧，说明没有发现回环
     * 那么把当前关键帧添加到关键帧数据库，然后返回False */
    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF); //【2】添加当前关键帧到数据库，用于回环检测和重定位
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    /**
     * 对于每个回环候选者，还需要检查与之前的回环候选者的一致性
     * 首先将候选帧扩展为一个共视组，即候选帧及其相邻关键帧组成一个候选组
     * 如果候选组与上一帧的一致群组至少共享一个帧，则认为他们是一致的，该群组的连续一致性加一，并用候选组更新上一帧的一致群组
     * 如果连续一致性大于3，即连续三个关键帧周期都被列为回环候选者，则将当前候选关键帧作为“足够一致候选者”输出
     */
    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);

    /* 遍历候选关键帧 */
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        /* 取候选关键帧及其所有连接（相邻）关键帧作为候选组 */
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        /* 遍历上一帧的一致群组，检查一致群组中是否有哪个组含有当前候选组的关键帧，即是一致的，属于同一个组 */
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                /* 找到与当前候选组一致的组，当前的一致性加一 */
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                /* 将当前候选组添加到当前一致群组 */
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                /* 如果当前的一致性大于阈值3，则将当前候选组添加到“足够一致候选者”，作为回环检测结果 */
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    /* 用当前一致群组更新上一帧的一致群组 */
    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF); //【3】添加当前关键帧到数据库，用于回环检测和重定位

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

/**
 * @brief 计算出当前帧与回环候选帧的空间变换Rt并搜索更多的匹配特征点
 * 
 * 基于该Rt在当前帧和回环帧及其相邻帧之间找出更多的匹配特征点，如果特征点足够多，则接受回环
 * 
 * 1. 遍历DetectLoop生成的回环候选帧，在当前帧和候选帧之间通过ORB方法寻找匹配的特征点，如果特征数量足够多，就为之创建一个Sim3Solver
 * 2. 轮流对每个回环候选帧的Sim3Solver进行5次迭代求解
 *   a. 如果求解成功，剔除匹配特征点中的离点
 *   b. 基于求解的R和t，通过投影的方法，实现更精准的搜索，在当前帧和候选帧寻找更多匹配的特征点
 *   c. 进行Sim3优化，获得当前关键帧与回环候选帧之间的精确位姿
 *   d. 有任何一个候选帧优化后内点的数量不少于20个，则停止RANSACS迭代，该候选帧即最为最终的闭环关键帧
 * 3. 取得闭环关键帧及其相邻关键帧的所有Map点
 * 4. 基于优化后的位姿变换，从闭环Map点中搜索更多与当前关键帧匹配的特征点
 * 5. 如果匹配的特征点足够多则接受该闭环。
 * 
 * 所谓Sim3优化就是求解两帧之间位姿Rt的优化
 * 
 * @return true 成功
 * @return false 失败
 */
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    /* 为每个一致回环候选帧计算出一个Rt */
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    /* 遍历回环候选帧，在当前帧和候选帧之间寻找匹配的特征点，特征点数量足够多的话，就创建一个Sim3Solver */
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        /* 在当前帧和候选帧之间基于ORB的方法寻找匹配的特征点数量 */
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        /* 如果特征点数量少于20个则丢弃，否则创建一个Sim3Solver */
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    /* 轮流对每个候选者进行5次RANSAC迭代，直到其中一个成功或者全部失败 */
    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        /* 遍历回环候选帧，对其Sim3Solver进行5次迭代求解 */
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            /* 取得候选帧 */
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            /* 对候选帧的Sim3Solver进行5次迭代求解 */
            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            /* 如果求解成功 */
            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                /* 根据RANSAC迭代的结果，保留匹配特征点中的内点，剔除离点 */
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                /**
                 * 取到RANSAC迭代获得的R和t，通过投影的方法，实现更精准的搜索，寻找更多两帧间匹配的特征点
                 */
                cv::Mat R = pSolver->GetEstimatedRotation();    //候选帧pKF到当前帧mpCurrentKF的R
                cv::Mat t = pSolver->GetEstimatedTranslation(); //候选帧pKF到当前帧mpCurrentKF的t
                const float s = pSolver->GetEstimatedScale();   //候选帧pKF到当前帧mpCurrentKF的变换尺度s（s12）
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                /* OpenCV的Mat矩阵转成Eigen的Matrix类型 */
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);

                /* 进行Sim3优化，获得当前关键帧与回环候选帧之间的精确位姿 */
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                /* 如果优化后内点的数量不少于20个，则停止RANSACS迭代 */
                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true; //回环匹配成功
                    mpMatchedKF = pKF; // 记录下完成匹配的回环候选关键帧，即闭环关键帧
                    /* 得到从世界坐标系到候选帧的位姿变换 */
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    /* 得到从世界坐标系到当前帧的位姿变换，其中gScm是从候选帧到当前帧的位姿变换 */
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw); //转成OpenCV的Mat格式

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break; // 只要有一个候选帧通过了Sim3求解和优化，就跳出对其他候选帧的判断
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    /* 取得所有的闭环Map点：遍历闭环关键帧及其相邻关键帧的所有Map点，将Map点添加到mvpLoopMapPoints中 */
    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        /* 遍历关键帧的Map点 */
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                /* 把所有的Map点添加到mvpLoopMapPoints中 */
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId; //仅仅是为了避免重复添加Map点
                }
            }
        }
    }

    /* 基于给定的位姿变换，从闭环Map点中搜索更多与当前关键帧匹配的特征点，在CorrectLoop中会用到 */
    /* mvpCurrentMatchedPoints以当前关键帧的特征点序号为顺序的索引表，以序号索引匹配的Map点 */
    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    /* 如果匹配的特征点足够多，则接受该闭环 */
    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        /* 清除所有非当前闭环关键帧的记录，返回成功 */
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

/**
 * @brief 闭环
 * 
 * 1. 取得当前帧及其相邻帧
 * 2. 取得当前帧的校正前和校正后位姿
 * 3. 遍历当前帧及其相邻帧
 *   a. 以当前帧为中介，取得所有相邻帧的校正前和校正后位姿
 *   b. 对当前帧及其相邻帧的Map点坐标以及位姿进行校正
 *   c. 启动回环闭合：用匹配的闭环Map点替换当前帧的对应Map点
 * 4. 基于校正后的位姿，将闭环Map点融合到当前帧及其相邻帧，替换既有的Map点，实现闭环
 * 5. 遍历当前帧及其相邻帧，更新他们的相邻帧，引入闭环帧，取得更新后的相邻帧，从中删除闭环前的
 *    一级和二级相邻帧，就只剩下闭环相邻帧，用于本质图优化
 * 6. 进行本质图优化
 * 7. 在当前帧和闭环帧之间添加闭环边
 * 8. 启动一个新的线程进行全局BA，对所有的关键帧和所有的Map点进行BA优化
 * 9. 闭环结束，删除LocalMapper中的所有关键帧
 * 10. 记录最近一次完成闭环的关键帧
 */
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    /* 等待LocalMapper停止工作 */
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    /* 取得当前关键帧及其相邻帧 */
    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    /* 取得校正前后的当前帧位姿 */
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;   //校正后位姿
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();  //校正前位姿，从当前帧相机坐标系到世界坐标系

    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        /* 以当前帧为中介，取得所有相邻帧的校正前和校正后位姿 */
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            /* 取得pKFi的位姿，从世界坐标系到该帧的相机坐标系 */
            cv::Mat Tiw = pKFi->GetPose();

            /* 以当前帧为中介，获得校正后的从pKFi到世界坐标系的位姿变换 */
            if(pKFi!=mpCurrentKF)
            {
                /* 获得从当前帧到pKFi的相对变换 */
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                /* 获得校正后的从pKFi到世界坐标系的位姿变换 */
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            /* 获得校正前的从pKFi到世界坐标系的位姿变换 */
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        /* 对当前帧及其相邻帧的Map点坐标以及位姿进行校正 */
        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            /* 取得pKFi的所有Map点并遍历 */
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId) //防止重复校正
                    continue;

                /* 先将pMPi映射到未校正的pKFi相机坐标系，然后在反映射到校正后的世界坐标系 */
                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                /* 用校正后的世界坐标系坐标更新pMPi */
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId; //标识该点已被校正
                pMPi->mnCorrectedReference = pKFi->mnId; //标识基于pKFi被校正
                pMPi->UpdateNormalAndDepth(); //更新Map点的平均观测方向以及观测距离范围
            }

            /* 用校正后的位姿更新pKFi的位姿 */
            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            /* 根据共视关系更新当前帧与其他关键帧之间的连接 */
            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        /**
         * 启动回环闭合：用匹配的闭环Map点替换当前帧的对应Map点 
         */
        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i]; // 与当前帧匹配的闭环Map点
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i); // 当前帧原有的Map点
                if(pCurMP) //当前帧有对应的Map点
                    pCurMP->Replace(pLoopMP); // 完成替换
                else
                {
                    /* 当前帧没有对应的Map点，直接替换 */
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    /* 将当前帧添加到该闭环Map点的观测帧中 */
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    /* 更新该闭环Map点的独特描述符 */
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    /* 基于校正后的位姿，将闭环Map点融合到当前帧及其相邻帧，替换既有的Map点，实现闭环 */
    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    /* 遍历当前帧及其相邻帧，更新他们的相邻帧，引入闭环帧，取得更新后的相邻帧，从中删除闭环前的一级和二级相邻帧，就只剩下闭环相邻帧 */
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        /* 取得pKFi旧的相邻帧 */
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        /* 当前帧及其相邻帧的部分Map点替换成了闭环Map点，当前帧及其相邻帧的相邻帧必然发生变化，必须更新 */

        /* 更新pKFi的相邻帧 */
        // Update connections. Detect new links.
        pKFi->UpdateConnections();

        /* 取得pKFi更新后的相邻帧，其中应当包含闭环帧 */
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();

        /* 从pKFi新的相邻帧中删除闭环之前的二级相邻帧 */
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        /* 从pKFi新的相邻帧中删除闭环之前的一级相邻帧 */
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
        /* 剩下的就纯粹是由闭环得到的新的相邻帧 */
    }

    /* 进行本质图优化 */
    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    /* 通知地图发生重大变化 */
    mpMap->InformNewBigChange();

    /* 在当前帧和闭环帧之间添加闭环边 */
    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    /* 启动一个新的线程进行全局BA，对所有的关键帧和所有的Map点进行BA优化 */
    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    /* 闭环结束，删除LocalMapper中的所有关键帧 */
    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    /* 记录最近一次完成闭环的关键帧 */
    mLastLoopKFid = mpCurrentKF->mnId;   
}

/**
 * @brief 基于校正后的位姿，将闭环Map点融合到当前帧及其相邻帧，替换既有的Map点，实现闭环
 * 
 * 1. 将闭环Map点坐标投影变换到pKF帧像素坐标系，检查冲突并融合，没有冲突的直接添加，
 *    冲突的既有Map点通过vpReplacePoints输出，暂不替换。
 * 2. 冲突的Map点在加锁保护的前提下再进行替换。
 * 
 * @param CorrectedPosesMap 当前关键帧及其相邻帧的校正后位姿，从相机坐标系到世界坐标系
 */
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    /* 遍历当前关键帧及其相邻帧，将闭环Map点融合到当前关键帧及其相邻帧 */
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        /* 取得校正后的位姿 */
        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        /* 将闭环Map点坐标投影变换到pKF帧像素坐标系，检查冲突并融合
         * 没有冲突的直接添加，冲突的既有Map点通过vpReplacePoints输出，暂不替换 */
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        /* 冲突的Map点在加锁保护的状态下再进行替换 */
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
