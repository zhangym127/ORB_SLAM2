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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyPoint数据类型

using namespace std;

namespace ORB_SLAM2
{

/** @brief 初始化跟踪线程
  * @param pSys SLAM系统
  * @param pVoc 词典
  * @param pFrameDrawer 帧抽屉
  * @param pMapDrawer Map抽屉
  * @param pMap Map
  * @param pKFDB 关键帧数据库
  * @param strSettingPath 设置文件路径
  * @param sensor 传感器类型
  */
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    /* 每一帧提取的特征点数量：1000 */
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    /* 建立图像金字塔时的缩放比例：1.2 */
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    /* 建立图像金字塔的层数：8 */
    int nLevels = fSettings["ORBextractor.nLevels"];
    /* 提取fast特征点的默认阈值：20 */
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    /* 如果默认阈值提取不出足够fast特征点，则使用最小阈值：8 */
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    /* 创建左侧ORB特征提取器 */
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    /* 如果是双目，则创建右侧ORB特征提取器 */
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    /* 对于单目，创建初始化时使用的特征提取器，主要区别是特征点数量翻了一番，变成了2000 */
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

/** @brief 处理一帧单目图像
  * @param im [in] 单目图像
  * @param timestamp [in] 图像对应的时间戳
  * @return 该帧图像对应的相机位姿，主要用于viewer等显示观察之用
  */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    /* 图像转成灰度图 */
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    /* 构造当前Frame：mCurrentFrame，提取特征点，畸变校正，分配到64×48的网格mGrid中 */
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    /* 进行帧到帧、帧到本地Map的跟踪以获得当前帧的精确位姿mTcw */
    Track();

    /* 返回当前帧对应的相机位姿，主要用于viewer等显示观察之用 */
    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief 进行帧到帧、帧到本地Map的跟踪以获得当前帧的精确位姿
 * 
 * 1. 单目初始化，找到当前帧的位姿，建立初始的单目地图；
 * 2. 进行帧到帧的跟踪，获得当前帧的位姿
 *   a. 如果上一帧有位姿，则基于运动模型，参照上一帧的位姿获得当前帧的位姿
 *   b. 否则，跟踪参考关键帧的Map点，通过优化获得当前帧的精确位姿
 *   c. 如果定位丢失，则进行重定位
 * 3. 在帧到帧跟踪的基础上，与本地Map进行匹配以获得更加精确的位姿，本质上是扩大范围与相邻的多个关键帧进行特征点匹配，并进行位姿优化
 * 4. 如果与本地Map的匹配效果良好：
 *   a. 获得当前帧相对于上一帧的位姿变换，作为新的运动模型
 *   b. 删除那些观测为零的Map点，即无效的Map点
 *   c. 删除跟踪运动模型时添加的临时Map点，仅仅用于提高双目或RGBD的跟踪效果，用完即删除
 *   d. 如果必要则创建一个新的关键帧，并将关键帧添加到LocalMapper
 * 5. 如果初始化之后很快就丢失定位则进行系统复位，并返回
 * 6. 记录当前帧的位姿信息，用于轨迹复现
 */
void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        /**
         * 步骤1：初始化，对于单目：
         * 1.找到初始帧和当前帧之间匹配的特征点
         * 2.基于匹配的特征点找到当前帧的位姿
         * 3.建立初始的单目地图，将mState设置成OK
         */
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else /* 步骤2：跟踪 */
    {
        // System is initialized. Track Frame.
        bool bOK;

        /**
         * mbOnlyTracking等于false表示正常VO模式（有地图更新），等于true表示用户手动选择定位模式
         * 在viewer中有个开关menuLocalizationMode，有它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
         */
        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                /* 检查上一帧中的MapPoints是否被替换，如果有替换则更新 */
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                /**
                 * 如果有运动模型，即mVelocity不为空，则优先使用运动模型，参考上一帧的位姿，
                 * 获得当前帧的Map点和精确位姿；否则，跟踪参考关键帧，通过BoW匹配的方法找到
                 * 当前帧的Map点以及位姿。BoW匹配的方法显然需要耗费更多的算力，成功的概率也
                 * 不高，因此肯定是优先使用运动模型。
                 */

                /**
                 * 运动模型是空的或刚完成重定位
                 * mnLastRelocFrameId记录上一次重定位的帧id
                 */
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    /**
                     * 跟踪参考关键帧的Map点，通过优化获得当前帧的精确位姿
                     */
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    /* 跟踪运动模型，对照上一帧的位姿，获得当前帧的精确位姿 */
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else /* 定位丢失 */
            {
                /**
                 * 对当前帧进行重定位，找到当前帧的位姿 
                 * 1.在关键帧数据库中检索与当前帧相似的候选关键帧
                 * 2.进行PnP匹配获得当前帧位姿的估计值，
                 * 3.然后对当前帧的位姿进行优化，找到当前帧的精确位姿
                 * 4.进行多轮循环和迭代在候选关键帧中找到最优的结果
                 */
                bOK = Relocalization();
            }
        }
        else // 纯定位模式，地图不更新
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST) /* 定位丢失 */
            {
                /* 对当前帧进行重定位，找到当前帧的位姿 */
                bOK = Relocalization();
            }
            else /* 跟踪状态正常 */
            {
                /**
                 * mbVO是mbOnlyTracking为true时的才有的一个变量
                 * mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
                 * mbVO为true表明此帧匹配了很少的MapPoints，少于10个，进入VO模式
                 */
                if(!mbVO) /* 上一帧的Map点数量较多，跟踪正常 */
                {
                    // In last frame we tracked enough MapPoints in the map

                    /* 跟踪运动模型或参考帧，获得当前帧的精确位姿 */
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else /* 上一帧的Map点数量较少，跟踪异常，VO模式 */
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    /**
                     * 下面计算两次当前帧的位姿，一次是基于运动模型，一次是重定位。如果重定位成功则使用重定位的结果，
                     * 否则继续使用运动模型的结果，维持VO模式。
                     */

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        /* 跟踪运动模型，对照上一帧的位姿，获得当前帧的精确位姿 */
                        bOKMM = TrackWithMotionModel();
                        /* 暂存跟踪运动模型的结果，如果下面的重定位失败则恢复跟踪运动模型的结果 */
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    /* 对当前帧进行重定位，找到当前帧的位姿 */
                    bOKReloc = Relocalization();

                    /* 重定位失败，但是跟踪运动模型成功，则恢复跟踪运动模型的结果，维持VO模式 */
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO) /* 当前帧的Map点数量较少，跟踪异常 */
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    /* 该点不是离点，表明被当前帧可靠观测到 */
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc) /* 重定位成功，直接使用重定位的结果，退出VO模式 */
                    {
                        mbVO = false;
                    }

                    /* 跟踪正常 */
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        /* 记录当前帧对应的参考帧 */
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        /**
         * 通过上一帧或者关键帧匹配得到初始的位姿后，现在与本地Map进行匹配以获得更加精确的位姿，
         * 本质上是扩大范围与相邻的多个关键帧进行特征点匹配，并进行位姿优化
         */
        if(!mbOnlyTracking) /* 地图更新模式 */
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else /* 纯定位模式下，如果mbVO为false，即匹配的Map点较多，则也进行本地Map的跟踪和优化 */
        {
            /* mbVO为true表示当前帧匹配到很少的Map点，也就无法建立对应的本地Map，因此不进行本地Map的跟踪。
             * 一旦系统重定位了，再使用本地Map进行跟踪 */
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        /* 根据本地Map跟踪的结果确定跟踪状态，定位是否丢失，是否需要重定位 */
        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        /* 如果跟踪良好，则检查是否需要插入一个关键帧 */
        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            /* 更新运动模型，即mVelocity，即当前帧相对于上一帧的位姿变化量，前提是上一帧也是有位姿的 */
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                /* 取上一帧位姿（世界坐标系到相机坐标系的变换）的逆 */
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                /* 取得当前帧对上一帧位姿的相对位姿，作为运动模型 */
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            /* 删除那些观测为零的Map点 */
            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            /* 删除跟踪运动模型时添加的临时Map点，仅仅用于提高双目或RGBD的跟踪效果，用完即删除 */
            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            /* 如果必要，则创建一个新的关键帧，并将关键帧添加到LocalMapper */
            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            /* 剔除当前帧的所有Map点中的离点 */
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        /* 如果初始化之后很快就丢失定位则进行系统复位 */
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        /* 更新当前帧的参考关键帧 */
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    /* 记录当前帧的位姿信息，用于轨迹复现 */
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        /* 计算并记录当前帧相对于参考关键帧的位姿变换 */
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        /* 记录当前参考关键帧 */
        mlpReferences.push_back(mpReferenceKF);
        /* 记录当前帧的时间戳 */
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        /* 记录当前是否丢失定位 */
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        /* 如果当前帧的位姿丢失，则直接记录上一帧的位姿等信息 */
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

/**
 * @brief 单目跟踪初始化，初始帧和参考帧获得位姿，建立Map，开始优化
 * 
 * 1.找出当前帧和参考帧之间的匹配特征点
 * 2.基于匹配特征点找出当前帧相对于参考帧的位姿R和t，还原出3D空间点
 * 3.将位姿R和t添加到参考帧和当前帧
 * 4.创建初始的单目地图
 */
void Tracking::MonocularInitialization()
{
    /* 创建初始化程序 */
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            /* 得到初始化的第一帧，初始化需要两帧 */
            mInitialFrame = Frame(mCurrentFrame);
            /* 记录最近的一帧 */
            mLastFrame = Frame(mCurrentFrame);
            /* 将当前帧中的特征点作为初始的匹配上的特征点 */
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            /* 创建初始化程序，设置Ransac标准差1.0（即1个像素），最大迭代次数200 */
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            /* 将初始匹配向量清空 */
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        /* 如果当前帧的特征点数大于100，则获得单目初始化的第二帧，开始初始化
         * 如果特征点数量小于100，则重新生成初始化程序 
         * 因此只有连续两帧特征点数大于100才能启动初始化*/
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // 在mCurrentFrame和mInitialFrame中找到匹配的特征点对，保存在mvIniMatches中
        // mvbPrevMatched保存前一帧匹配的特征点
        // mvIniMatches存储mInitialFrame,mCurrentFrame之间匹配的特征点
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 如果两帧之间匹配的点太少，则重新生成初始化程序
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation 当前帧的旋转R
        cv::Mat tcw; // Current Camera Translation 当前帧的平移t
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches) 指示mvIniMatches中通过三角化以及重投影检验的特征点对

        /* 基于当前帧和参考帧之间的匹配点，找出当前帧相对于参考帧的旋转R和平移t，还原出3D空间点 */
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                /* 剔除mvIniMatches中未通过三角化检验的特征点对儿 */
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            /* 将位姿R和t添加到到当前帧和初始帧中 */

            // Set Frame Poses
            /* 设置初始帧的位姿：单位矩阵 */
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            /* 设置当前帧的位姿：Rt */
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F); //当前的相机位姿变换
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            /* 创建初始的单目地图 */
            CreateInitialMapMonocular();
        }
    }
}

/**
 * @brief 创建初始的单目地图
 * 
 * 1.分别为初始帧和当前帧创建关键帧，关键帧的一个主要特征是相互之间有连接（边），有权重描述
 * 2.创建关键帧的词袋
 * 3.将关键帧插入到Map中
 * 4.将与两帧对应的3D空间点添加到Map
 *  a.将3D空间点添加到关键帧，建立从关键帧到Map点的映射
 *  b.将关键帧添加到3D空间点的观察中，建立从Map点到关键帧的映射
 *  c.更新3D空间点的独特描述符
 *  d.更新Map点的平均观测方向以及观测距离范围
 * 5.更新关键帧之间的连接关系、权重，并根据权重建立树形结构
 * 6.进行BA优化，更新关键帧的位姿，更新3D空间点的坐标
 * 7.单目传感器无法恢复真实的深度，将关键帧位姿和3D空间点坐标归一化
 * 8.更新LocalMapper等变量
 * 9.将mState设置成OK，表征初始化完成
 */
void Tracking::CreateInitialMapMonocular()
{
    /* 创建关键帧 */
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    /* 创建关键帧的词袋 */
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    /* 将关键帧插入到Map中 */
    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    /* 遍历每一个通过检验的特征点对儿，将对应的3D空间点添加到Map */
    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //为每一个3D空间点创建一个Map点
        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        /* 用3D空间点的坐标、关键帧、Map初始化Map点 */
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        /* 建立从关键帧到Map点的映射关系 */
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        /* 为Map点添加该关键帧对应的观察 */
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        /* 更新独特描述符 */
        pMP->ComputeDistinctiveDescriptors();
        /* 更新Map点的平均观测方向以及观测距离范围 */
        pMP->UpdateNormalAndDepth();

        /* 填写当前帧的Map点以及离点结构 */
        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        /* 添加Map点到Map */
        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    /**
     * 更新关键帧间的连接关系，对于一个新创建的关键帧都会执行一次关键连接关系更新
     * 在关键帧之间建立边，每个边有一个权重，边的权重是两帧之间拥有的共视3D点的个数
     */
    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    /* BA优化，迭代20次。BA优化只和关键帧的位姿（顶点）、Map点的坐标（顶点）以及从
     * Map点到关键帧的投影即特征点（边）相关，与上面创建的关键帧之间的连接没有关系 */
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    /* 取得初始帧的场景深度中值，参数2表示取中值深度 */
    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    /* 单目传感器无法恢复真实的深度，这里将关键帧位姿和点云坐标归一化到1 */

    /* 用中值深度对当前帧的平移向量进行归一化 */
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    /* 用中值深度对所有的Map点坐标进行归一化 */
    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    /* 将初始帧和当前帧作为关键帧插入到LocalMapper中，LocalMapper将启动工作*/
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    /* 更新若干变量 */
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    /* 初始化完成，将状态设置为OK */
    mState=OK;
}

/**
 * @brief 检查上一帧中的MapPoints是否被替换
 * 关键帧在local_mapping和loopclosure中存在融合(fuse)Map点的情况，
 * 由于这些Map点被改变了，就需要检查并更新Map点
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 跟踪参考关键帧的Map点，获得当前帧的精确位姿
 * 
 * 1. 基于当前帧的BRIEF描述符计算出BoW向量
 * 2. 在当前帧和参考帧之间进行ORB匹配，找出匹配的Map点，作为当前帧的Map点
 * 3. 将上一帧的位姿作为当前帧的初始位姿，进行位姿优化，获得当前帧的精确位姿
 * 4. 根据优化的结果将Map点中的离点剔除掉
 * 5. 剔除离点后，如果Map点总数大于10则返回true
 * 
 * @return true 剔除离点后Map点的总数＞10
 */
bool Tracking::TrackReferenceKeyFrame()
{
    /* 基于当前帧的BRIEF描述符计算出BoW向量 */
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    /* 在当前帧和参考帧之间进行ORB匹配，如果匹配点足够多则启动PnP优化 */
    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    /* 通过词袋BoW提高匹配速度 */
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    /* 将当前帧与参考关键帧匹配的点作为当前帧的Map点 */
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    /* 将上一帧的位姿作为当前帧位姿的初始值 */
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    /* 开始当前帧的位姿优化 */
    Optimizer::PoseOptimization(&mCurrentFrame);

    /* 剔除离点 */
    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

/**
 * @brief 对于双目或RGBD，根据特征点的深度值为上一帧构造一些临时map点
 * 
 * 1. 基于参考帧的位姿以及相对位姿更新上一帧的位姿
 * 2. 选取一些比较近的特征点，构造为临时map点
 */
void Tracking::UpdateLastFrame()
{
    /* 基于参考帧的位姿以及相对位姿更新上一帧的位姿 */
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    /* 如果上一帧是关键帧，或者是单目，则退出 */
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    /* 得到上一帧有深度值的特征点 */
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    /* 对深度值排序，从小到大 */
    sort(vDepthIdx.begin(),vDepthIdx.end());

    /**
     * 取距离比较近的点，构造临时Map点
     * 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
     * 之后在CreateNewKeyFrame之前会全部删除
     */
    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            /* 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除 */
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/**
 * @brief 跟踪运动模型，对照上一帧的位姿，获得当前帧的精确位姿
 * 
 * 1. 根据上一帧的位姿估计当前帧的位姿
 * 2. 在当前帧中找到与上一帧Map点匹配的特征点，将Map点添加到当前帧
 * 3. 基于找到的Map点，对当前帧的位姿进行优化，获得精确位姿
 * 4. 基于优化后的位姿，剔除当前帧Map点中的离点
 * 5. 如果当前帧的总的Map点数少于10个，则将mbVO设置为TRUE
 * 
 * @return true 跟踪成功
 * @return false 跟踪失败
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    /* 对于双目或RGBD，根据特征点的深度值为上一帧构造一些临时map点 */
    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    /* 根据上一帧的位姿估计当前帧的位姿，mVelocity是最近一次前后帧位姿增量 */
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    /* 清空当前帧的map点 */
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    /**
     * 在当前帧找到与上一帧Map点匹配的特征点，并将该Map点添加到当前帧
     * 1.遍历上一帧的所有Map点，将其通过3D-2D投射到当前帧，获得当前帧上P点坐标
     * 2.取得当前帧上P点附件的特征点，找到与上一帧Map点描述符距离最近的特征点
     * 3.如果描述符距离小于阈值，则将该Map点添加到当前帧
     * th阈值主要用于确定P点周围的搜索半径，单位是像素
     */
    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    /* 如果找到的匹配点太少，则搜索半径×2，重新搜索 */
    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    /* 基于找到的Map点，对当前帧的位姿进行优化，获得精确位姿 */
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    /* 剔除所有的离点 */
    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    /* 如果只定位不建图 */
    if(mbOnlyTracking)
    {
        /* 如果当前帧的总的Map点数少于10个，则将mbVO设置为TRUE */
        mbVO = nmatchesMap<10;
        /* 与上一帧的匹配点数大于20就认为跟踪成功 */
        return nmatches>20;
    }

    /* Map点的数量至少10个才算成功 */
    return nmatchesMap>=10;
}

/**
 * @brief 通过与本地Map的匹配获得更加精确的位姿，本质上是扩大范围与相邻的多个关键帧进行特征点匹配以及位姿优化
 * 
 * 1. 更新本地Map，将所有与当前帧存在共视关系的关键帧及其相邻帧以及父子关键帧的Map点更新到本地Map
 * 2. 通过ORB匹配的方法搜素与当前帧中特征点匹配的本地Map点，添加为当前帧的Map点
 * 3. 基于更新后的当前帧Map点，对当前帧的位姿进行优化，获得精确位姿
 * 4. 统计位姿优化后内点的数量，如果高于阈值则认为跟踪本地Map成功
 * 
 * @return true 跟踪本地Map成功
 * @return false 跟踪本地Map失败
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    /**
     * 关于相机的位姿已经有一个初步的估计，并且有一些可以跟踪的地图点。
     * 检索本地地图，并尝试查找与本地地图中的点匹配的项。
     */

    /* 更新本地Map点 */
    UpdateLocalMap();

    /* 通过ORB匹配的方法搜素与当前帧中特征点匹配的本地Map点，添加为当前帧的Map点 */
    SearchLocalPoints();

    /* 基于更新后的当前帧Map点，对当前帧的位姿进行优化，获得精确位姿 */
    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    /* 位姿优化会将Map点划分成内点和离点两类，统计内点的数量 */
    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                /* 该点不是离点，表明被当前帧可靠观测到 */
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    /* 判断是否跟踪成功 */
    // Decide if the tracking was succesful

    /* 如果刚刚进行过重定位，则内点的数量应不少于50个 */
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    /* 其他情况下内点的数量应不少于30个 */
    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/**
 * @brief 确认是否需要一个新的关键帧
 * 
 * 不需要的情况：
 * 1. 只跟踪不建图
 * 2. 本地Map被回环检测所冻结
 * 3. 自从上次重定位以来不足1秒钟
 * 
 * 需要的情况：
 * 1. 跟踪比较弱：与本地Map匹配的内点数量少于能够跟踪到的Map点
 * 2. 本地Map处于空闲状态
 * 3. 自上次重定位以来超过1秒钟
 * 
 * 如果本地地图处于空闲状态，则插入关键帧，否则发送一个信号中断BA
 * 
 * @return true 需要
 * @return false 不需要
 */
bool Tracking::NeedNewKeyFrame()
{
    /* 如果只跟踪不建图，则不需要新增关键帧 */
    if(mbOnlyTracking)
        return false;

    /* 如果本地Map被回环检测所冻结也不能新增关键帧 */
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    /* 获取地图中关键帧的总数 */
    const int nKFs = mpMap->KeyFramesInMap();

    /* 如果自从上次重定位以来还没有经过足够数量的帧，则也不新增关键帧
     * mMaxFrames是图像的帧率，相当于是要求距离上次重定位或者初始化至少1秒钟以上 */
    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    /* 获取参考关键帧中观测（能够同时看到该点的关键帧的）数量大于nMinObs的Map点的数量，即能够跟踪到的点 */
    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    /* 查询局部地图是否繁忙，是否能够接受新的关键帧 */
    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    /* 对于双目或RGBD：统计近距离内点和离点的总数 */
    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    /* 近距离内点的数量不足100个，并且近距离离点的数量大于70个 */
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    /* 超过1秒钟未插入关键帧 */
    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    /* 本地地图处于空闲状态 */
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    /* 双目或RGBD情况下，与本地Map匹配的内点数量少于能够跟踪到的Map点的25%，或者近距离内点的数量不足，即跟踪比较弱 */
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    /* (与本地Map匹配的内点数量少于能够跟踪到的Map点 或者 近距离内点的数量不足) 并且 与本地Map匹配的内点数量多于15个 */
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        /* 如果本地地图处于空闲状态，则插入关键帧，否则发送一个信号中断BA */
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            /* 中断BA */
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                /* 对于双目或RGBD， 中断BA后，如果本地Map中的关键帧数量不足3个则插入关键帧 */
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true; 
                else
                    return false;
            }
            else /* 对于单目，中断BA后暂不插入关键帧 */
                return false;
        }
    }
    else
        return false;
}

/**
 * @brief 创建一个关键帧，并将关键帧添加到LocalMapper
 * 
 * 1. 将当前帧构造为关键帧
 * 2. 将新建的关键帧设置为当前帧的参考关键帧
 * 3. 对于双目和RGBD，构造一些Map点，确保至少有100个Map点
 * 4. 将新创建的关键帧插入LocalMapper，【注意】这是初始化完成后LocalMapper的唯一输入
 * 5. 更新上一关键帧指针变量为当前关键帧
 */
void Tracking::CreateNewKeyFrame()
{
    /* 设置标志防止本地Map被冻结 */
    if(!mpLocalMapper->SetNotStop(true))
        return;

    /* 将当前帧构造为关键帧 */
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    /* 将新建的关键帧设置为当前帧的参考关键帧 */
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    /* 对于双目和RGBD，构造一些Map点，确保至少有100个Map点 */
    if(mSensor!=System::MONOCULAR)
    {
        /* 更新当前帧与世界坐标系之间的旋转、平移和相机光心坐标等变量 */
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            /* 对深度值排序，从小到大 */
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                /* 如果Map点不存在或者没有观测（即没有别的帧观测到该Map点）则需要新建Map点 */
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                /* 创建新的Map点 */
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    /* 将新创建的关键帧插入LocalMapper，LocalMapper将启动工作，【注意】这是初始化完成后LocalMapper的唯一输入 */
    mpLocalMapper->InsertKeyFrame(pKF);

    /* 取消防止LocalMapper被冻结的标志 */
    mpLocalMapper->SetNotStop(false);

    /* 更新上一关键帧为当前关键帧 */
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 通过ORB匹配的方法搜素与当前帧中特征点匹配的本地Map点，添加为当前帧的Map点
 * 
 * 1. 遍历当前帧已有的Map点，标记为不参与当前帧特征点的ORB匹配
 * 2. 遍历mvpLocalMapPoints中所有的本地Map点，如果该点在当前帧的视野内，则标记为参与当前帧特征点的ORB匹配
 * 3. 通过ORB匹配的方法，搜索当前帧中与mvpLocalMapPoints中Map点匹配的特征点，如果找到则将Map添加为该特征点对应的Map点
 */
void Tracking::SearchLocalPoints()
{
    /* 遍历当前帧的既有的Map点，标记为不参与当前帧特征点的ORB匹配 */
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                /* 剔除无效点 */
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                /* 观察到该Map点的帧数+1 */
                pMP->IncreaseVisible();
                /* 标记该点被当前帧所观察到 */
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                /* 标记该点不参与当前帧特征点的ORB匹配，因为已经匹配过 */
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    /* 遍历所有的本地Map点，如果该点在当前帧的视野内，则标记为参与当前帧特征点的ORB匹配 */
    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        /* 已经被当前帧观察到的Map点不再判断是否能被当前帧观察到 */
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;

        /* 判断Map点是否在当前帧的视野内，并填充Map点的成员变量用于跟踪，0.5是余弦值，
         * 表示当前帧对该点的观测方向与该Map点的平均观测方向的夹角应小于60°。
         * 如果Map点在当前帧的视野内，则将mbTrackInView设置为True。 */
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            /* 观测到该点的帧数加一 */
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    /* 通过ORB匹配的方法，搜索当前帧中与mvpLocalMapPoints中Map点匹配的特征点，如果找到则将Map添加为该特征点对应的Map点。
     * 只有mbTrackInView被设置为True的Map点才纳入本地Map点的ORB匹配 */
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        
        /* 如果最近两帧之内进行过重定位，则需要进行更大范围的搜索，设定5倍的放大系数 */
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        /* 通过ORB匹配的方法搜索搜索当前帧中与mvpLocalMapPoints中Map点匹配的特征点，如果找到则将Map添加为该特征点对应的Map点 */
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

/**
 * @brief 更新本地Map
 * 
 * 1. 将mvpLocalMapPoints中的点设置为参考Map点
 * 2. 将所有与当前帧存在共视关系的关键帧及其相邻帧以及父子关键帧全都加入到mvpLocalKeyFrames变量中
 * 3. 将mvpLocalKeyFrames中新添加的所有关键帧的所有Map点添加到mvpLocalMapPoints
 */
void Tracking::UpdateLocalMap()
{
    /* 将本地Map点设置为Map的参考点，这在创建初始地图的时候已经干过一次 */
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update

    /** 
     * 更新本地关键帧，即更新mvpLocalKeyFrames变量，将所有与当前帧存在共视关系的关键帧及其相邻帧，
     * 以及子关键帧和父关键帧全都加入到mvpLocalKeyFrames变量中。*/
    UpdateLocalKeyFrames();

    /**
     * 更新本地Map点，即更新mvpLocalMapPoints变量，将mvpLocalKeyFrames中新添加的所有关键帧的所
     * 有Map点添加到mvpLocalMapPoints。 */
    UpdateLocalPoints();
}

/**
 * @brief 更新本地Map点，即更新mvpLocalMapPoints变量
 * 
 * 将mvpLocalKeyFrames中新添加的所有关键帧的所有Map点添加到mvpLocalMapPoints。
 * 
 */
void Tracking::UpdateLocalPoints()
{
    /* 清空 */
    mvpLocalMapPoints.clear();

    /* 遍历mvpLocalKeyFrames中新添加的关键帧 */
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        /* 取得关键帧下所有的Map点 */
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        /* 遍历所有的Map点，添加到mvpLocalMapPoints中，并标记该Map点的跟踪参考帧为当前帧 */
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 更新本地关键帧，即更新mvpLocalKeyFrames变量
 * 
 * 1. 遍历当前帧的每一个Map点，获取Map的所有观察，获得与当前帧存在共视关系的关键帧，并统计每个关键帧能够看到的Map点数
 * 2. 所有与当前帧存在共视关系的关键帧都被添加到mvpLocalKeyFrames，并记录具有最大共视关系（共享最多Map点）的关键帧pKFmax
 * 3. 遍历2中新添加的关键帧：
 *   a. 将既有关键帧相邻（拥有最多共视关系）的前10个关键帧添加到mvpLocalKeyFrames
 *   b. 将既有关键帧的所有子关键帧添加到mvpLocalKeyFrames
 *   c. 将既有关键帧的父关键帧添加到mvpLocalKeyFrames
 *   d. 关键帧的总数达到80个则停止添加
 * 4. 将2中记录的关键帧pKFmax设置为当前帧的参考关键帧mpReferenceKF
 * 
 */
void Tracking::UpdateLocalKeyFrames()
{
    /* 遍历当前帧的每一个Map点，获取Map的所有观察，从中统计每个关键帧能够看到的Map点数 */
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    /* 记录哪一个关键帧与当前帧共享最多的Map点 */
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    /* 记录所有与当前帧存在共视关系的关键帧 */
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    /* 所有与当前帧存在共视关系的关键帧都被包含在本地map中，同时也检查与哪一个关键帧共享最多的Map点 */
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    /* 同时也包含一些与既有关键帧相邻的关键帧，达到80个就可以了 */
    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        /* 取得并遍历与该关键帧有连接（最多共视关系）的前10个关键帧，全部添加到本地关键帧中 */
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        /* 取得并遍历该关键帧的所有子关键帧，全部添加到本地关键帧中 */
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        /* 取得该关键帧的父关键帧，添加到本地关键帧中 */
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    /* 将与当前帧拥有最多共视关系的帧设置为当前帧的参考关键帧 */
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @brief 在定位丢失的情况下，对当前帧进行重定位，找到当前帧的位姿
 * 
 * 1. 计算当前帧的BoW向量
 * 2. 在关键帧数据库中检索与当前帧相似的候选关键帧
 * 3. 对于每一个候选关键帧:
 *   a. 进行与当前帧的ORB匹配，如果匹配点足够多，则设置一个PnP solver
 *   b. 通过EPnP算法估计当前帧的位姿，并给出内点及其数量
 *   c. 基于估计的位姿，对当前帧的位姿进行优化，获得精确位姿
 *   d. 如果优化后的内点不足50个，则将关键帧的Map点通过3D-2D投射到当前帧的方法搜索更多的匹配特征点，并再次进行优化
 * 4. 如果没有哪个关键帧与当前帧匹配的内点数量超过50个，则重复第3步，进行多次迭代
 * 
 * @return true 重定位成功
 * @return false 重定位失败
 */
bool Tracking::Relocalization()
{
    /* 计算当前帧特征点的BoW向量 */
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    /**
     * 跟踪丢失的情况下进行重定位
     * 在关键帧数据库中检索与当前帧相似的候选关键帧，以实现重定位
     */
    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    /**
     * 对于每一个候选关键帧，进行与当前帧的ORB匹配，如果匹配点足够多，则设置一个PnP solver
     */
    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            /* 通过BoW搜索关键帧和当前帧之间匹配的特征点，返回匹配的Map点 */
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else /* 如果匹配的特征点数量大于等于15个，则用当前帧初始化一个PnPsolver，并添加到vpPnPsolvers中 */
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    /* 开展多轮迭代，直到与某个关键帧匹配的内点数量大于50个 */
    while(nCandidates>0 && !bMatch)
    {
        /* 遍历每个候选关键帧 */
        for(int i=0; i<nKFs; i++)
        {
            /* 跳过被丢弃的关键帧 */
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            /* 通过EPnP算法估计当前帧的位姿，并给出内点及其数量 */
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                /* 将PnP算法获得的位姿作为当前帧位姿的估计值 */
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                /* 将匹配的Map点添加到当前帧，并剔除离点 */
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                /* 对当前帧的位姿进行优化，获得精确位姿 */
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                /* 从当前帧的Map点中剔除离点 */
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                /* 如果内点不足，则通过3D-2D的投影匹配搜索更多的匹配Map点，并再次进行优化 */
                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    /* 搜索当前帧与关键帧匹配的Map点并添加到当前帧 */
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        /* 如果Map点的总数超过50个，则进行再次优化，获得当前帧的精确位姿 */
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        /** 如果内点仍然不足，以更小的范围进行再次搜素。由于当前帧的位姿已经优化过，
                         * 因此有希望在更小的搜索范围下找到更多的匹配点 */
                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            /* 搜索半径从10个像素减少到3个，BRIEF描述符距离阈值从100降到64 */
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                /* 再次优化 */
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                /* 提出离点 */
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }

                /* 如果内点的数量大于50个则停止迭代，否则重复上述过程 */
                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        /* 将当前帧标记为最近一次重定位帧 */
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
