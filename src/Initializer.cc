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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

/** @brief 单目SLAM的初始化程序
  * @param ReferenceFrame 参考帧
  * @param sigma 标准差
  * @param iterations 随机采样一致迭代次数
  */
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    /* mvKeys1记录参考帧的特征点 */
    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}

/** @brief 单目SLAM的初始化
 * 
 * 1.初始化mvKeys2、mvMatches12、mvbMatched1、mvSets等变量
 * 2.每次RANSAC迭代选择8对匹配特征点，总共迭代mMaxIterations次，存放在mvSets中
 * 3.启动线程并行计算基本矩阵F和单应矩阵H，以及对应的分值
 * 4.从基本矩阵F或单应矩阵H中恢复姿态R和位置t
 * 
 * @param CurrentFrame 当前帧
 * @param vMatches12 当前帧与初始帧(参考帧)之间匹配的特征点
 * @param R21 当前帧的旋转
 * @param t21 当前帧的平移
 * @param vP3D 
 * @param vbTriangulated
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2

    /* mvKeys2记录当前帧的特征点 */
    mvKeys2 = CurrentFrame.mvKeysUn;

    /* mvMatches12记录当前帧和参考帧匹配上的特征点对，根据当前帧的特征点数量进行初始化 */
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());

    /* mvbMatched1记录参考帧中的每个特征点是否有匹配的特征点 */
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    /* mvMatches12的实际大小是当前帧和参考帧之间匹配上的特征点数量 */
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    /* vAllIndices中记录完整的索引表[0~N) */
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    /**
     * 每次RANSAC迭代选择8对匹配特征点，总共迭代mMaxIterations次，存放在mvSets中,
     * mMaxIterations = 200，
     * mvSets是一个200×8的二维数组
     */
    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    /* 迭代mMaxIterations次，每次随机选择8对匹配特征点，存放在mvSets中 */
    for(int it=0; it<mMaxIterations; it++)
    {
        /* 先获得完整的索引表 */
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            /* 将被选中的索引项从索引表中删除，确保不会被重复选中 */
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    /* 启动线程并行计算基础矩阵F和单应矩阵H，以及对应的分值 */
    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF; //基本矩阵F和单应矩阵H对应的分值
    cv::Mat H, F;

    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    /* 等待线程完成计算 */
    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    /* 计算单应矩阵分值在总分值中的比例 */
    // Compute ratio of scores
    float RH = SH/(SH+SF);

    /** 
     * 从基础矩阵F或单应矩阵H中恢复姿态R和位置t 
     * 1. 如果H的占比超过40%，则从H矩阵恢复，否则从F矩阵恢复
     * 2. 参数1.0表示进行checkRT时恢复的3D点视差角阈值
     * 3. 参数50表示满足checkRT检测的3D点个数（checkRT时会恢复3D点）
     */
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}

/**
 * @brief 计算单应矩阵
 * 
 * @param vbMatchesInliers 
 * @param score 
 * @param H21 
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // 获得当前帧和参考之间匹配点的数量
    // Number of putative matches
    const int N = mvMatches12.size();

    // 分别对参考帧和当前帧的特征点坐标进行归一化，使坐标均值为0，偏差均值（一阶绝对矩）为1，归一化矩阵分别为T1和T2
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv(); //T2inv是T2的逆矩阵

    // 初始化最优结果变量
    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        /* 计算单应矩阵 */
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);

        /* 恢复原有的均值和尺度 */
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        /* 记录分数最高的结果及其分数 */
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**
 * @brief 计算基础矩阵
 * 
 * @param vbMatchesInliers[out] 利用基础矩阵重映射后精度达标的特征点对儿
 * @param score[out] 最高评分
 * @param F21[out] 最高评分对应的基础矩阵
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // 分别对参考帧和当前帧的特征点坐标进行归一化，使坐标均值为0，偏差均值（一阶绝对矩）为1，归一化矩阵分别为T1和T2
    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // 初始化最高评分等
    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // 迭代mMaxIterations次，每次取出事先保存在mvSets中的8对儿匹配特征点，计算基础矩阵
    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        /* 计算基础矩阵 */
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        /* 恢复原有的尺度和均值 */
        F21i = T2t*Fn*T1;

        /* 遍历所有特征点对儿，检查基础矩阵的效果，并累计所有满足偏差阈值的分值 */
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        /* 记录分数最高的基础矩阵 */
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**
 * @brief 根据输入的8对儿特征点计算单应矩阵
 * 
 * |x'|     | h1 h2 h3 ||x|
 * |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
 * |1 |     | h7 h8 h9 ||1|
 * 使用DLT(direct linear tranform)求解该模型
 * x' = a H x 
 * ---> (x') 叉乘 (H x)  = 0
 * ---> Ah = 0
 * A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
 *     |-x -y -1  0  0  0 xx' yx' x'|
 * 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解
 * 
 * 注意其中的叉乘，矩阵A来自x'叉乘Hx的结果。
 * 
 * @param vP1 归一化后的8个特征点，来自参考帧
 * @param vP2 归一化后的8个特征点，来自当前帧
 * @return cv::Mat 返回单应矩阵H
 */
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

/**
 * @brief 根据输入的8对儿特征点计算基础矩阵
 * 
 * 1. 利用8对特征点坐标构造矩阵A，根据对极约束写成Af=0的形式，其中f是向量形式的基础矩阵
 * 2. 利用SVD分解求解线性方程组Af=0：A分解成u,w,v，v的最后一列即w中最小的奇异值所对应的列向量即为解
 * 3. 将f从向量还原成3×3的本质矩阵F
 * 4. 根据基础矩阵F秩为2的特性，再次进行SVD分解，并强制将w的第三个值设置为0，再将u,w,v还原成基础矩阵
 * 
 * @param vP1 归一化后的8个特征点，来自参考帧
 * @param vP2 归一化后的8个特征点，来自当前帧
 * @return cv::Mat 返回基础矩阵F
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    //构造对极约束
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    // 对A(8×9)进行SVD分解，获得u(8×8)、w(8×9)、vt(9×9)三个子矩阵
    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 取v的最后一列，即vt的最后一行，获得本质矩阵F的初步解，还原成3×3矩阵
    // row(m)表示取矩阵的第m行，col(n)表示取矩阵的第n列
    // reshape(c, m)表示将矩阵改成c个通道、m行的矩阵，如果c=0，表示维持通道不变
    cv::Mat Fpre = vt.row(8).reshape(0, 3); 

    // 对本质矩阵F的初步解再次进行SVD分解，获得u(3×3)、w(3×3)、vt(3×3)三个子矩阵
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 根据本质矩阵秩为2的的约束，将第三个奇异值设置为0
    w.at<float>(2)=0;

    // 将u、w、vt重新组合成本质矩阵F
    return  u*cv::Mat::diag(w)*vt;
}

/**
 * @brief 检查单应矩阵，获得对应的评分
 * 
 * @param H21[in] 单应矩阵
 * @param H12[in] 单应矩阵
 * @param vbMatchesInliers[out] 记录重映射后偏差小于阈值的匹配点对儿
 * @param sigma[in] 标准差，数值等于1
 * @return float 当前单应矩阵的评分
 */
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    /* 基于卡方检验计算出的阈值（假设测量有一个像素的偏差） */
    const float th = 5.991;

    /* 标准差平方的倒数 */
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        /**
         * 将图像2中的特征点p2单应变换成图像1中的p1'
         * |u1|   |h11inv h12inv h13inv||u2|
         * |v1| = |h21inv h22inv h23inv||v2|
         * |1 |   |h31inv h32inv h33inv||1 |
         */
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        /* 计算p1和p1'的距离的平方 */
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        /* 计算卡方检验值，并与两个自由度的阈值比较，如果偏大则放弃，否则累计当前分值 */
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 检查基础矩阵，获得对应的评分
 * 
 * 检查的方法是利用基础矩阵F将图像1上的特征点x1转换成图像2上的极线l2，然后计算图像2上的特征点x2到l2的距离，获得偏差，
 * 卡方检验值=（偏差/方差）^2，然后与卡方检验中概率为95%时一个自由度的阈值3.84进行比较，仅当小于该阈值时满足要求，
 * 评分等于卡方检验中概率为95%时两个自由度的阈值5.99-(卡方检验值)^2，偏差越大，分值越低，偏差越小，分值越高。
 * 
 * @param F21[in] 待检查的基础矩阵F
 * @param vbMatchesInliers[out] 记录重映射后偏差小于阈值的匹配点对儿
 * @param sigma[in] 标准差，数值等于1
 * @return float 当前基础矩阵的评分
 */
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    /* 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）*/
    const float th = 3.841;
    const float thScore = 5.991;

    /* 标准差平方的倒数 */
    const float invSigmaSquare = 1.0/(sigma*sigma);

    /* 遍历所有匹配的特征点对儿 */
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        /**
         * 利用基础矩阵和特征点x1计算出图像2上的极线l2：Fx1=l2
         * F是基础矩阵，x1是图像1上的特征点，x2是图像2上对应的特征点，l2是图像2上的极线，x2应该落在l2上。
         */
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        /* 求点到直线的距离的平方，单位是像素。理想情况下，x2落在l2上，点到直线的距离等于0 */
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        /* 计算卡方检验值，并与一个自由度的阈值比较，如果偏大则放弃，否则累计当前分值 */
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        /**
         * 利用基础矩阵和特征点x2计算出图像1上的极线l2：Fx2=l1
         * F是基础矩阵，x2是图像2上的特征点，x1是图像1上的特征点，l1是图像1上的极线，x1应该落在l1上。
         */
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        /* 记录该对特征点是否是内点 */
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 从基础矩阵F中恢复姿态R和位置t
 * 
 * 
 * 
 * @param vbMatchesInliers[in] 重映射后偏差小于阈值的匹配点对儿
 * @param F21[in] 基础矩阵
 * @param K[in] 内参矩阵K
 * @param R21[out] 姿态R
 * @param t21[out] 位置t
 * @param vP3D[out] 
 * @param vbTriangulated[out] 
 * @param minParallax[in] 进行checkRT时恢复的3D点视差角阈值，数值是1.0
 * @param minTriangulated[in] 满足checkRT检测的3D点个数（checkRT时会恢复3D点），数值是50
 * @return true 成功的分解出R和t
 * @return false 分解失败
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    /* 将基础矩阵F转成本质矩阵E */
    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    /* 从本质矩阵中分解出位姿R和t */
    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    /* 对四个可能的解进行检查：[R1,t],[R1,-t],[R2,t],[R2,-t] */
    // Reconstruct with the 4 hypotheses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

/**
 * @brief 从单应矩阵H中恢复姿态R和位置t
 * 
 * 利用Faugeras的SVD分解法从H中分解出R和t
 * 
 * @param vbMatchesInliers 重映射后偏差小于阈值的匹配点对儿
 * @param H21[in] 单应矩阵
 * @param K[in] 内参矩阵K
 * @param R21[out] 姿态R
 * @param t21[out] 位置t
 * @param vP3D[out] 
 * @param vbTriangulated[out] 
 * @param minParallax[in] 进行checkRT时恢复的3D点视差角阈值，数值是1.0
 * @param minTriangulated[in] 满足checkRT检测的3D点个数（checkRT时会恢复3D点），数值是50
 * @return true 成功的分解出R和t
 * @return false 分解失败
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    /* 将单应矩阵H从像素坐标变换成相机坐标系的矩阵A */
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    /* 对矩阵A进行SVD分解 */
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    /* 取得奇异值，d1>d2>d3 */
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    /* 如果奇异值之间的差异很小，则直接返回false */
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

/**
 * @brief 
 * 
 * @param kp1 来自第一帧图像的特征点
 * @param kp2 来自第二帧图像的特征点
 * @param P1 第一帧图像的投影矩阵
 * @param P2 第二帧图像的投影矩阵
 * @param x3D 
 */
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

/**
 * @brief 对vector中的坐标进行归一化
 * 
 * @param vKeys[in] 待归一化的特征点
 * @param vNormalizedPoints[out] 完成归一化的特征点
 * @param T 
 */
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    /* 求所有点的坐标均值meanX和meanY，即质心 */
    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        /* 从坐标点上减去均值或质心坐标，使所有点的x和y坐标均值为0 */
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        /* 求扣除均值后的坐标绝对值之和，即偏差之和 */
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    /* 求所有点的偏差均值 */
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    /* 求偏差均值的倒数 */
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    /* 对所有点的x坐标和y坐标进行缩放，使其偏差均值为1，或者说一阶绝对矩为1 */
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

/**
 * @brief 检查姿态R和位置t的质量
 * 
 * @param R[in] 姿态或旋转R
 * @param t[in] 位置或平移t
 * @param vKeys1 参考帧的特征点
 * @param vKeys2 当前帧的特征点
 * @param vMatches12[in] 当前帧与参考帧匹配的特征点索引
 * @param vbMatchesInliers[in] 重映射后偏差小于阈值的匹配特征点索引
 * @param K[in] 内参矩阵
 * @param vP3D[out] 
 * @param th2[in] 
 * @param vbGood[out] 
 * @param parallax[out] 
 * @return int 
 */
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    /* 得到第一帧图像的投影矩阵P1，并以第一帧相机的光心O1作为世界坐标系原点 */
    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    /* 第一帧相机的光心在世界坐标系下的坐标O1 */
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    /* 得到第二帧图像的投影矩阵P2 */
    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    /**
     * 第二帧相机的光心在世界坐标系下的坐标
     * R和t是从第一帧到第二帧的运动，先平移后旋转，那么光心O2在世界坐标系下的坐标就是-R^t*t
     */
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

/**
 * @brief 从本质矩阵E中分解出位姿R和t
 * 
 * 分解本质矩阵E将得到四组解：[R1,t],[R1,-t],[R2,t],[R2,-t]
 * 
 * @param E[in] 本质矩阵 
 * @param R1[out] 姿态1
 * @param R2[out] 姿态2
 * @param t[out] 位移
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    /* 对本质矩阵进行SVD分解 */
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    /* 取U的最后一列作为t，并且对t进行归一化 */
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    /* 构造沿Z轴旋转90度的旋转矩阵W */
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    /* R1=UWVt */
    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    /* R2=UWtVt */
    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
