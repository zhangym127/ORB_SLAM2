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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

/**
 * @brief 创建关键帧数据库，关键帧数据库的主要用途是回环检测和重定位
 * 
 * 关键帧数据库的核心数据结构mvInvertedFile是以Word序号为索引，具有相同Word的关键帧list为元素的一个嵌套向量容器
 * 只要关键帧含有某个Word，则该关键帧就会被添加到该Word序号索引的关键帧列表中
 * 关键帧会被多次添加到不同Word索引的列表中
 * 
 * @param voc Word总数，Word的总数是固定的
 */
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

/**
 * @brief 添加关键帧到关键帧数据库
 * 
 * 【注意】数据库中的所有关键帧都来自LoopCloser，只有LoopCloser会添加关键帧到数据库
 * 
 * 关键帧数据库的核心数据结构mvInvertedFile是以Word序号为索引，具有相同Word的关键帧列表为元素的一个向量容器
 * 只要关键帧含有某个Word，则该关键帧就会被添加到该Word序号索引的关键帧列表中
 * 关键帧会被多次添加到不同Word索引的列表中
 * 
 * @param pKF 待添加的关键帧
 */
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

/**
 * @brief 从关键帧数据库中删除某个关键帧
 * 
 * @param pKF 待删除的关键帧
 */
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

/**
 * @brief 在关键帧数据库中寻找候选的回环关键帧
 * 
 * 寻找回环关键帧的关键在于关键帧数据库，从数据库中找到与当前关键帧既有很高的相似度又不相邻的关键帧作为检测到的回环关键帧。
 * 为了确保回环检测的可靠性，还要取候选帧的10个相邻帧，进一步检查是否同样具有较高的相似度。
 * 
 * 1. 在数据库中检索有共享Word的关键帧，并统计该关键帧共享的Word数量，但是要将与当前关键帧有连接关系（即相邻）的关键帧排除在外
 * 2. 取共享Word数量较多的关键帧，如果ORB相似度高于参考分值minScore，则作为候选关键帧
 * 3. 取候选帧的前10个最佳相邻帧，计算累计得分，取最高得分的75%作为最后的保留阈值，筛选出最高得分的组及最佳关键帧
 * 4. 返回最佳关键帧
 * 
 * @param pKF 当前关键帧
 * @param minScore 参考分值
 * @return vector<KeyFrame*> 候选的回环关键帧
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    /* 回环关键帧不可以是相邻帧，因此取得与当前关键帧有连接关系（相邻）的所有关键帧，用于甄别 */
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        /* 遍历当前关键帧的所有Word，在数据库中检索有共享Word的关键帧，并记录共享的Word数量 */
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            /* 从关键帧数据库中拿到含有当前Word的所有关键帧 */
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            /* 遍历这些含有相同Word的关键帧 */
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnLoopQuery!=pKF->mnId) // mnLoopQuery的唯一作用是防止重复添加候选帧
                {
                    pKFi->mnLoopWords=0;

                    /* 如果含有相同Word的关键帧和当前帧没有连接关系，即不是相邻帧 */
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        /* 将该帧标记为候选帧 */
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++; //该帧与当前关键帧共享的Word数量加一
            }
        }
    }

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    /* 只比较那些与当前帧有足够多的共享Word的关键帧 */
    // Only compare against those keyframes that share enough words

    /* 找出与当前帧有最多共享Word的关键帧，记录共享Word数量的最大值 */
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    /* 取最大值的80%作为下限 */
    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    /* 遍历所有的候选关键帧 */
    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        /* 只处理共享Word数量高于下限的关键帧 */
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            /* 获得该关键帧与当前关键帧的相似度 */
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

            /* 如果分数高于参考值，则作为进一步候选帧添加到lScoreAndMatch */
            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    /* 单单计算当前帧和某一关键帧的相似性是不够的，这里将与关键帧相连（权值最高，共视程度最高）的前十个关键帧归为一组，计算累计得分 */

    /* 遍历所有的候选关键帧 */
    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        /* 取得与该帧具有最佳共视关系的10个相邻帧 */
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi; //默认取候选关键帧为最佳关键帧

        /* 遍历这10个相邻帧 */
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            /* 如果相邻帧也在候选帧之列，并且共享Word数量高于下限 */
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                /* 累加相邻帧的相似度分数 */
                accScore+=pKF2->mLoopScore;
                /* 记录具有最高分值的相邻帧 */
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2; //如果候选帧的相邻帧分数更高，则取相邻帧为最佳关键帧
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        /* 针对每个候选关键帧，记录对应的组得分和具有最高分值的相邻帧 */
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        /* 记录候选关键帧中的最高组得分 */
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    /* 取最高组得分的75%作为最后的保留阈值 */
    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    /* 将高于阈值的候选关键帧保留下来，作为最终的回环候选关键帧 */
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}

/**
 * @brief 在关键帧数据库中寻找与给定的F帧相似的关键帧
 * 
 * 1. 从关键帧数据库中提取所有与当前帧共享某个Word的关键帧
 * 2. 统计每个关键帧与当前帧共享的Word数量，取最大的共享Word数量的80%作为阈值
 * 2. 筛选出共享Word大于阈值的关键帧，计算与当前帧的BoW相似度分值，记录相似度分值及其关键帧
 * 3. 对筛选出的关键帧，将关键帧及其具有最多连接关系（最佳共视）的前10个关键帧与当前帧的BoW相似度分支累加起来，记录累加相似度分值及其关键帧
 * 4. 以最高累加分数的75%为阈值，返回所有高于这个阈值的关键帧
 * 
 * @param F 当前帧
 * @return vector<KeyFrame*> 与当前帧相似的关键帧 
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    /* 找到所有与当前帧共享Word的关键帧 */
    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        /* 遍历当前帧的每一个Word */
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            /* 提取所有包含该Word的关键帧 */
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            /* 遍历所有共享Word的关键帧，更新关键帧中重定位相关的字段 */
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    /* 统计所有共享Word的关键帧中具有的最多共享Word数，用来决定阈值 */
    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    /* 最低阈值是最大共享Word数的80% */
    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    /* 遍历具有共享Word的关键帧，选择共享Word数大于最低阈值的关键帧，计算与当前帧的BoW相似度分数，记录相似度分数及其关键帧 */
    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        /* 选择共享Word数大于最低阈值的关键帧，计算与当前帧的BoW相似度分数，记录相似度分数及其关键帧 */
        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si; //注意这里将BoW分数记录在关键帧的mRelocScore字段中
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    /**
     * 遍历BoW相似度分数及其关键帧
     * 仅仅计算关键帧与当前帧的BoW相似度是不够的，下面将关键帧及其具有最多连接关系（最佳共视）的前10个关键帧归为一组，
     * 通过累加BoW相似度分数获得组分数。最后通过累加后的组分数筛选出具有最高相似度的关键帧。
     */
    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        /* 对于一个具有较高BoW相似度的关键帧*/
        KeyFrame* pKFi = it->second;
        /* 取到与该关键帧具有最多连接关系（具有最佳共视）的前10个关键帧，其实也就是相邻的帧 */
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        /* 把关键帧及其10个相邻帧与当前帧的BoW相似度分数累加起来，并记录最佳累加分数 */
        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            /* 仅当pKF2也是具有共享Word的候选帧时才能贡献分数 */
            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        /* 记录累加分数及其关键帧 */
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    /* 以最高累加分数的75%为阈值，返回所有高于这个阈值的关键帧 */
    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF; //防止重复添加的标识变量
    vector<KeyFrame*> vpRelocCandidates; //高于阈值的关键帧
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
