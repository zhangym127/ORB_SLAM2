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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

/**
 * @brief 搜索当前帧与vpMapPoints中匹配的Map点并添加到当前帧，只有mbTrackInView被设置为True的Map点才纳入匹配
 * 
 * 1. 对于vpMapPoints中的每一个Map点
 *   a.只有mbTrackInView被设置为True的Map点才纳入匹配
 *   b.以Frame::isInFrustum()中Map点3D-2D投影获得的像素坐标为中心，r为半径，返回当前帧指定范围内的特征点
 *   c.取得Map点的BRIEF描述符
 *   d.对于b中取到的每一个特征点，计算与Map点的描述符距离，记录最优和次优距离
 * 2. 如果最优结果相对于次优结果的优势不明显则跳过，否则将该Map点添加为最优匹配特征点对应的Map点
 * 3. 返回匹配的特征点数量，或新添加的Map点数量
 * 
 * @param F[in] 当前帧 
 * @param vpMapPoints 待匹配的Map点
 * @param th[in] 施加于搜索半径的系数 
 * @return int 返回匹配的特征点数量
 */
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];

        /* 只有mbTrackInView被设置为True的Map点才纳入匹配 */
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        /* 获得估计的Map点图像金字塔层级 */
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        /* 基于当前帧对该Map点的观测方向与该点的平均观测方向的夹角确定搜索半径 */
        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        /* 搜索半径乘以缩放系数 */
        if(bFactor)
            r*=th;

        /* 以Frame::isInFrustum()中3D-2D投影获得的像素坐标为中心，r为半径，返回当前帧指定范围内的特征点 */
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        /* 获得当前Map点的BRIEF描述符 */
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        /* 遍历当前帧中找到的特征点，找到与关键帧Map点最匹配的特征点 */
        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            /* 如果该特征点已经有对应的Map点及其观测，则跳过 */
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            /* 双目或RGBD时，需要确保右图的对应点也在搜索半径内 */
            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            /* 取得该特征点对应的描述符 */
            const cv::Mat &d = F.mDescriptors.row(idx);

            /* 计算两个描述符的距离 */
            const int dist = DescriptorDistance(MPdescriptor,d);

            /* 更新最优匹配结果，使用bestDist2和bestLevel2记录次优结果 */
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2) /* 更新次优匹配结果 */
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            /* 如果最优结果相比次优结果的优势并不明显则跳过 */
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            /* 添加该Map点到当前帧 */
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    /* 返回匹配的特征点数量 */
    return nmatches;
}

/**
 * @brief 根据视角确定搜索半径，单位是像素
 * 
 * 如果夹角小于3.6度则返回2.5，否则返回4.0
 * 
 * @param viewCos 该Map点的观测方向与该点的平均观测方向的夹角的Cos值
 * @return float 搜索半径，单位是像素
 */
float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    /* 如果夹角小于3.6度则返回2.5，否则返回4.0 */
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}

/**
 * @brief 计算特征点kp2到极线l2的距离是否小于阈值
 * 
 * 1. 通过特征点kp1和基本矩阵F计算出极线l2
 * 2. 计算特征点kp2到极线l2的距离
 * 3. 检查距离是否小于给定的阈值
 * 
 * @param kp1 特征点1
 * @param kp2 特征点2
 * @param F12 基本矩阵
 * @param pKF2 关键帧2
 * @return true 小于阈值
 * @return false 大于阈值
 */
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    /* 通过kp1和基本矩阵F计算出极线l2 */
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    /**
     * 计算kp2特征点到极线l2的距离
     * (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)
     */
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    /* 层级越高，尺度越大 */
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

/**
 * @brief 在当前帧和参考关键帧之间寻找匹配的特征点
 * 
 * FIXME: 具体的寻找方法有待进一步分析
 * 
 * @param pKF[in] 参考关键帧
 * @param F[in] 当前帧
 * @param vpMapPointMatches[out] 匹配的特征点
 * @return int 匹配的特征点数量
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];

                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;

                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                    const int dist =  DescriptorDistance(dKF,dF);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<=TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMapPointMatches[bestIdxF]=pMP;

                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/**
 * @brief 根据给定的位姿变换，从闭环Map点中搜索更多与当前关键帧匹配的特征点
 * 
 * 1. 遍历所有的闭环Map点，将每个点投影到当前帧，获得投影点及其搜索半径，在当前帧中搜索位于该区域的素有特征点；
 * 2. 遍历所有找到的特征点，与Map点进行ORB匹配，当阈值满足要求时认为匹配成功，记录下匹配的特征点
 * 
 * @param pKF 当前关键帧
 * @param Scw 从世界坐标系到当前帧的位姿变换
 * @param vpPoints 所有的闭环Map点，包括闭环关键帧及其相邻帧的所有Map点
 * @param vpMatched 闭环关键帧与当前关键帧之间匹配的特征点
 * @param th 重投影偏差，10个像素
 * @return int 返回匹配的特征点数量
 */
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    /* 记录已经匹配的点，剔除没有匹配的Null项 */
    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    /* 遍历所有的闭环Map点，投影并进行匹配 */
    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos(); //取得闭环Map点的世界坐标

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw; //转到当前帧的相机坐标系

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0) //深度必须为正
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy; //转到当前帧的像素坐标

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        /* Map点在当前帧的景深必须在合理范围内 */

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        /* Map点在当前帧的视角和该Map点的平均视角的夹角必须小于60度 */

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        /* 根据Map点的景深估计所在图像金字塔层级 */
        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        /* 根据图像金字塔层级确定搜索半径 */
        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        /* 以(u,v)为中心，以radius为半径，返回当前帧指定范围内的所有特征点 */
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        /* 取得Map点的描述符 */
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        /* 遍历所有找到的特征点 */
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            /* 忽略图像金字塔层级与Map点不符的特征点 */

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            /* 取得该特征的描述符 */
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            /* 计算该特征点与Map点描述符的距离 */
            const int dist = DescriptorDistance(dMP,dKF);

            /* 记录最优匹配结果 */
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        /* 如果匹配结果小于阈值，则认为匹配成功 */
        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    /* 返回匹配的特征点数量 */
    return nmatches;
}

/**
 * @brief 在第1帧和第2帧之间寻找匹配的特征点
 * 
 * FIXME: 特征点的匹配方法有待进一步分析
 * 
 * @param F1 第1帧
 * @param F2 第2帧
 * @param vbPrevMatched 第1帧与其前一帧之间匹配的特征点，如果第1帧是初始帧，则是第1帧全部的特征点 
 * @param vnMatches12 第1、2帧之间匹配的特征点
 * @param windowSize 
 * @return int 匹配的特征点数量
 */
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    /* 初始化输出变量 */
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

/**
 * @brief 搜索两个关键帧中匹配的特征点
 * 
 * 通过ORB的方法提取两个关键帧中匹配的特征点，要求提取到的特征点必须有对应的Map点
 * 
 * @param pKF1[in] 关键帧1
 * @param pKF2[in] 关键帧2
 * @param vpMatches12[out] 匹配的特征点
 * @return int 
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    /* 分别取得两个关键帧的特征点、特征向量、描述符、Map点 */
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    /**
     * 将属于同一节点(特定层)的特征点进行ORB特征匹配
     * FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
     * f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
     */
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    /* 遍历pKF1和pKF2中的node节点 */
    while(f1it != f1end && f2it != f2end)
    {
        /* 确保f1it和f2it属于同一个node节点 */
        if(f1it->first == f2it->first)
        {
            /* 遍历f1it节点下所有的特征点 */
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                /* 确保该特征点对应的Map点存在 */
                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                /* 取得该特征点对应的描述符 */
                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                /* 遍历f2it节点下所有的特征点 */
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    /* 确保该特征点对应的Map点存在 */
                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    /* 取得该特征点对应的描述符 */
                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    /* 计算两个特征点描述符的距离 */
                    int dist = DescriptorDistance(d1,d2);

                    /* 记录最佳匹配结果 */
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                /* 检查匹配特征点的旋转一致性 */
                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        /* 记录匹配的特征点 */
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    /* 检查匹配特征点的旋转一致性 */
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/**
 * @brief 搜索关键帧之间满足对极约束的匹配点对儿
 * 
 * 1. 将帧1光心转到帧2像素坐标系，获得帧2中极点的坐标，为检查是否满足对极约束做准备
 * 2. 将帧1中的每个特征点与帧2中的所有特征点依次进行ORB匹配，通过匹配的再检查是否满足对极约束
 * 3. 将通过ORB匹配且满足对极约束的点对儿记录下来，返回匹配的特征点对儿数量
 * 
 * @param pKF1 关键帧1
 * @param pKF2 关键帧2
 * @param F12  基础矩阵
 * @param vMatchedPairs 匹配点对儿 
 * @param bOnlyStereo 在双目和rgbd情况下，要求特征点在右图存在匹配
 * @return int 返回匹配的点对儿数量
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter(); /* 获得帧1光心在世界坐标系下的坐标 */
    cv::Mat R2w = pKF2->GetRotation();    /* 获得从世界坐标系到帧2的旋转 */
    cv::Mat t2w = pKF2->GetTranslation(); /* 获得从世界坐标系到帧2的平移 */
    cv::Mat C2 = R2w*Cw+t2w;              /* 获得帧1光心在帧2相机坐标系下的坐标 */
    /* 将帧1光心从帧2相机坐标系转成帧2像素坐标系，即获得帧1在帧2中对应的极点坐标 */
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    /**
     * 将属于同一节点(特定层)的特征点进行ORB特征匹配，匹配的点再检查是否满足对极约束
     * FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
     * f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
     */
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    /* 遍历pKF1和pKF2中的node节点 */
    while(f1it!=f1end && f2it!=f2end)
    {
        /* 如果f1it和f2it属于同一个node节点 */
        if(f1it->first == f2it->first)
        {
            /* 遍历f1it指向node节点下的所有特征点 */
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                /* 取得特征点索引值 */
                const size_t idx1 = f1it->second[i1];
                
                /* 取到索引值对应的Map点 */
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                /* 该特征点已经有对应的Map点，不需要新建 */
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                /* 如果右侧图像中有索引值对应的特征点，表示是双目 */
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                /* 取得索引值对应的特征点 */
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                /* 取得索引值对应的描述符 */
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                /* 遍历f2it指向node节点下的所有特征点 */
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    /* 计算两个描述符距离 */
                    const int dist = DescriptorDistance(d1,d2);
                    
                    /* 如果描述符距离大于阈值则放弃 */
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    /* 取得索引值对应的特征点 */
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        /* 计算特征点kp2到极点的平面距离，如果距离太近，表明kp2对应的Map点距离基线很近，放弃！ */
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    /* 检查特征点kp2到极线l2的距离是否小于阈值，小于阈值的记录下来 */
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                /* 如果找到ORB特征匹配且满足对极约束的特征点对儿 */
                if(bestIdx2>=0)
                {
                    /* 将匹配的特征点对儿记录下来 */
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    /* 检查特征方向 */
                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    /* 检查特征方向 */
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    /* 匹配的特征点对儿从vMatches12转存到vMatchedPairs */
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    /* 返回匹配的特征点数量 */
    return nmatches;
}

/**
 * @brief 将当前帧的Map点与关键帧pKF中的Map点进行融合
 * 
 * 1. 将当前帧的Map点投影到关键帧pKF，取得投影点附近的所有特征点；
 * 2. 通过检查重投影偏差以及ORB匹配找到与当前帧Map点最匹配的特征点；
 * 3. 如果该特征点已经有Map点，则在该Map点和当前帧Map点中选择具有最大观测值的Map点作为融合后的Map点
 * 4. 如果该特征点还没有Map点，则将当前帧Map点添加到该特征点
 * 
 * @param pKF 待融合的关键帧
 * @param vpMapPoints 当前帧的Map点
 * @param th 判断是否是重复Map点的阈值，默认值是3个像素
 * @return int 返回融合的Map点数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    /* 遍历所有的Map点 */
    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        /* 如果当前Map点也在关键帧pKF中，即在pKF中能看到该Map点，则不需要合并 */
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        /* 取到当前Map点的世界坐标，转到关键帧pKF的相机坐标系 */
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        /* 再转成关键帧pKF的像素坐标(u,v) */
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        /* 检查该Map点在pKF帧中的景深是否在正常范围内 */
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        /* 检查该Map点在pKF帧中的视角是否在60度范围内 */
        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        /* 根据特征点到相机关心的距离估计所在的图像金字塔层级 */
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        /* 根据所在图像金字塔层级确定特征点的搜索半径 */
        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        /* 取得关键pKF上，以(u,v)为中心，以radius为半径的圆内的所有特征点，作为候选特征点 */
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        /* 取得当前Map点的描述符 */
        const cv::Mat dMP = pMP->GetDescriptor();

        /* 遍历找到的候选特征点 */
        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            /* 过滤层级不匹配的特征点 */
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            /* 检测候选特征点与当前Map点对应特征点的重投影偏差 */
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                /* 基于卡方检验计算出的阈值 */
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            /* 取得候选特征点的描述符 */
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            /* 计算两个特征点的描述符距离 */
            const int dist = DescriptorDistance(dMP,dKF);

            /* 记录最匹配的特征点 */
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            /* 如果最匹配特征点已经有对应的Map点，则选择那个拥有最观测的Map点作为融合后的Map点 */
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else /* 如果最匹配特征点还没有对应的Map点，则将当前Map点添加到该特征点 */
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

/**
 * @brief 将闭环Map点与关键帧pKF中的Map点进行融合
 * 
 * 1. 将闭环Map点投影到关键帧pKF，取得投影点附近的所有特征点；
 * 2. 通过检查重投影偏差以及ORB匹配找到与当前帧Map点最匹配的特征点；
 * 3. 如果该特征点已经有Map点，则在该Map点记录在vpReplacePoint中
 * 4. 如果该特征点还没有Map点，则将闭环Map点添加到该特征点
 * 
 * @param pKF[in] 当前关键帧或其相邻帧
 * @param Scw[in] 该帧的校正后位姿，从世界坐标系到相机坐标系
 * @param vpPoints[in] 闭环Map点
 * @param th[in] 判断是否是重复Map点的阈值，默认值是4个像素
 * @param vpReplacePoint[out] 融合过程中需要替换的既有Map点
 * @return int 融合的Map点的数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    /* 取得pKF的所有Map点 */
    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    /* 遍历所有的闭环Map点 */
    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        /* 如果pKF已经包含pMP则略过，无需融合 */
        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        /**
         * 取得pMP在世界坐标系下的坐标，投影到pKF，获得像素坐标(u,v)
         */
        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        /* 确认投影后的点在pKF的图像范围内 */
        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        /* 确认pMP在pKF中的景深处于正常范围 */
        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        /* 确认pMP在pKF中的视角与pMP平均观测方向的夹角小于60° */
        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        /* 根据pMP在pKF中的景深估计所在图像金字塔层级 */
        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        /* 根据所在层级以及搜索范围阈值确定搜索半径 */
        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        /* 取得pKF中以(u,v)为中心，以radius为半径的圆内的所有特征点 */
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        /* 取得pMP的描述符 */
        const cv::Mat dMP = pMP->GetDescriptor();

        /* 遍历找到的候选特征点 */
        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            /* 确认所在层级基本相当 */
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            /* 取得特征点的描述符 */
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            /* 计算pMP和该特征点描述符的距离 */
            int dist = DescriptorDistance(dMP,dKF);

            /* 记录最佳匹配 */
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            /* 取得pKF中最佳匹配特征点对应的Map点 */
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                /* 有对应的Map点，则记录该Map点 */
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                /* 没有对应的Map点，直接将pMP添加到pKF */
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    /* 返回融合的点数 */
    return nFused;
}

/**
 * @brief 通过给定两帧间变换关系搜索更多的匹配特征点
 * 
 * 1. 遍历关键帧1的所有Map点，将Map点投影到关键帧2，在关键帧2中搜索匹配的特征点
 * 2. 遍历关键帧2的所有Map点，将Map点投影到关键帧1，在关键帧1中搜索匹配的特征点
 * 3. 如果1和2中搜索到的特征点能对应起来，则确认该特征点完全匹配，添加到输出结果中
 * 4. 在输入的匹配特征点的基础上添加新匹配的特征点
 * 
 * @param pKF1[in] 关键帧1 
 * @param pKF2[in] 关键帧2 
 * @param vpMatches12[inout] 输入已经匹配的特征点，输出更新后的匹配特征点，数量上只会增加，不会减少 
 * @param s12  
 * @param R12[in] 两个关键帧之间的旋转关系 
 * @param t12[in] 两个关键帧之间的平移关系 
 * @param th[in] 匹配的阈值，7.5个像素 
 * @return int 匹配的特征点数量
 */
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    /* 分别取得两个关键帧的Map点 */
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    /* 记录两个关键帧既有特征点的匹配情况 */
    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    /* 遍历关键帧1的所有Map点，在关键帧2中搜索匹配的特征点 */
    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        /* 略过已经匹配的Map点 */
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        /* 分别将Map点转到两个关键帧的相机坐标系 */
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        /* 将Map点在关键帧2中的坐标进一步转成像素(u,v) */
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        /* 确认(u,v)在帧2的图像范围内 */
        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        /* 确认Map点在关键帧2的正常景深范围内 */
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        /* 根据景深计算出图像金字塔的估计层级 */
        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        /* 根据图像金字塔层级确定图像的搜索半径，单位是像素 */
        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        /* 以(u,v)为中心，以radius为半径，返回关键帧2上指定范围内的所有特征点 */
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        /* 取得Map点对应的描述符 */
        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        /* 遍历在关键帧2上找到的特征点 */
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            /* 略过图像金字塔层级不匹配的特征点 */
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            /* 取得该特征点对应的描述符 */
            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            /* 计算Map点与该特征点的描述符距离 */
            const int dist = DescriptorDistance(dMP,dKF);

            /* 记录最佳匹配特征点 */
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        /* 记录关键帧1中的Map点在关键帧2中匹配的特征点 */
        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    /* 遍历关键帧2的所有Map点，在关键帧1中搜索匹配的特征点 */
    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        /* 记录关键帧2中的Map点在关键帧1中匹配的特征点 */
        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    /* 如果vnMatch1中记录匹配结果与vnMatch2能对应上，则确认该特征点完全匹配，添加到输出结果中 */
    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

/**
 * @brief 搜索当前帧与上一帧匹配的Map点并添加到当前帧
 * 
 * 1. 对于双目或RGBD，判断当前帧是前进还是后退，主要目的是缩小图像金字塔的搜索范围
 * 2. 将上一帧的Map点投射到当前帧，获得对应特征点坐标的估计值
 * 3. 以该坐标为中心，取得当前帧中搜索半径和图像金字塔层级范围内的特征点
 * 4. 遍历这些点，找到与上一帧Map点的BRIEF描述符距离最近的点及其距离
 * 5. 如果描述符距离满足阈值，则将该Map点添加到当前帧
 * 6. 进行旋转一致性检查，如果不一致，则将Map点从当前帧中删除
 * 
 * @param CurrentFrame[in] 当前帧
 * @param LastFrame[in] 上一帧
 * @param th[in] 确定搜索半径的阈值，单位是像素
 * @param bMono[in] 是否是单目
 * @return int 返回匹配的特征点数量
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    /* 从世界坐标系到当前帧相机坐标系的变换 */
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    /* 当前帧相机坐标系的坐标原点在世界坐标系下的坐标 */
    const cv::Mat twc = -Rcw.t()*tcw;

    /* 从世界坐标系到上一帧相机坐标系的变换 */
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    /* 当前帧相机坐标系的坐标原点在上一帧坐标系下的坐标 */
    const cv::Mat tlc = Rlw*twc+tlw;

    /* 对于双目或者RGBD，判断当前帧是前进还是后退 */
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    for(int i=0; i<LastFrame.N; i++)
    {
        /* 取上一帧的Map点 */
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                /* 将上一帧的Map点投射到当前帧的坐标系下x3Dc */
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

                /* 完成从3d-2d的投影，获得Map点对应的图像坐标(u,v) */
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                /* 检查图像坐标是否超出当前帧的图像边界 */
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                int nLastOctave = LastFrame.mvKeys[i].octave;

                /* 根据特征点所在的图像金字塔层级确定搜索半径 */
                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

                /* 找到当前帧中(u,v)附近的特征点 */
                if(bForward)
                    /* 前进时，距离变近，上一帧的特征点在当前帧需要更高的金字塔层级才能检测出来 */
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
                    /* 后退时，距离变远，上一帧的特征点在当前帧需要更低的金字塔层级才能检测出来 */
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
                    /* 对于单目无法判断深度，也就无法判断前进还是后退，只能扩大搜索范围 */
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                /* 取得上一帧Map点的描述符 */
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                /* 遍历当前帧中找到的特征点，找到与上一帧Map点最匹配的特征点 */
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    /* 如果该特征点已经有对应的Map点及其观测，则跳过 */
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    /* 双目或RGBD时，需要确保右图的对应点也在搜索半径内 */
                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        /* 求右侧对应点的x坐标 */
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    /* 取得该特征点对应的描述符 */
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    /* 计算两个描述符的距离 */
                    const int dist = DescriptorDistance(dMP,d);

                    /* 更新最优匹配结果 */
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    /* 添加该Map点到当前帧 */
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    /* 计算旋转一致性 */
                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    /* 如果旋转一致性检查不通过，则将该Map点从当前帧中删除 */
    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

/**
 * @brief 搜索当前帧与关键帧匹配的Map点并添加到当前帧
 * 
 * 1. 取得关键帧的所有Map点
 * 2. 将关键帧的Map点投射到当前帧，获得对应特征点坐标的估计值
 * 3. 以该坐标为中心，取得当前帧中搜索半径和图像金字塔层级范围内的特征点
 * 4. 遍历这些点，找到与关键帧Map点的BRIEF描述符距离最近的点及其距离
 * 5. 如果描述符距离满足阈值，则将该Map点添加到当前帧
 * 6. 进行旋转一致性检查，如果不一致，则将Map点从当前帧中删除
 * 
 * @param CurrentFrame[in] 当前帧
 * @param pKF[in] 关键帧
 * @param sAlreadyFound[in] 已知的共视Map点
 * @param th[in] 确定搜索半径的阈值，单位是像素
 * @param ORBdist[in] BRIEF描述符距离阈值 
 * @return int 返回匹配的特征点数量
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    /* 计算当前帧相机坐标系的坐标原点在世界坐标系下的坐标Ow */
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    /* 取得关键帧的所有Map点 */
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    /* 遍历关键帧的每一个Map点 */
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            /* 如果Map点是有效的，也不在已知的共视Map点中 */
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                /* 将Map点投射到当前帧下x3Dc */
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                /* 完成从3d-2d的投影，获得Map点对应的图像坐标(u,v) */
                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                /* 计算从Ow到x3Dw的向量PO，即世界坐标系下，从当前帧相机坐标原点到Map点坐标的向量 */
                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO); //计算PO向量的长度，即Map点在当前帧相机坐标系下的深度

                /* 取得观测到该Map点的最大和最小距离 */
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                /* Map点的深度必须在图像金字塔的范围内 */
                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                /* 根据Map点的深度估计所在的图像金字塔层 */
                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                /* 确定搜索半径，与所在的图像金字塔层相关 */
                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                /* 在指定的搜索半径内，找到当前帧中(u,v)附近的特征点 */
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                /* 获得关键帧Map点pMP的BRIEF描述符 */
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                /* 遍历当前帧中找到的特征点，找到与关键帧Map点最匹配的特征点 */
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    /* 将匹配的Map点添加到当前帧 */
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    /* 计算旋转一致性 */
                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    /* 如果旋转一致性检查不通过，则将该Map点从当前帧中删除 */
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/**
 * @brief 计算两个BRIFF描述符的距离
 * 
 * @param a[in] 描述符a
 * @param b[in] 描述符b
 * @return int 描述符之间的距离
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        /* 两个描述符进行异或运算，相同为0，不同为1 */
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
