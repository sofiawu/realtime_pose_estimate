//
//  cylinder_detection.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include "estimate.h"

#include "model.hpp"
#include "robust_matcher.hpp"
#include "utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2//core.hpp>
#include <opencv2/video/tracking.hpp>

using namespace posest;

const int num_keypoints = 800;

//static const double width = 1920, height = 1080;
/*static const double params_WEBCAM[] = { 0.8638 * width,   // fx
    1.1527 * height,  // fy
    0.5012 * width - 0.5,      // cx
    0.4975 * height - 0.5};    // cy
*/
static const double param_IPhone7P[] = { 1822.858, 1815.437, 956.428, 498.178 };
static const double dist_coeff[] = { 0.198096, -0.880351, -0.008620, 0.002268 };

static const int pnp_method = CV_ITERATIVE;
static const int iter_count = 500;
static const float reprojection_error = 3.0f;
static const double confidence = 0.95;

// Kalman Filter parameters
static const int minInliersKalman = 30;    // Kalman threshold updating


/////////////////////

static void InitKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);
static void UpdateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                               cv::Mat &translation_estimated, cv::Mat &rotation_estimated );
static void FillMeasurements( cv::Mat &measurements,
                             const cv::Mat &translation_measured, const cv::Mat &rotation_measured);

/////////////////////

int CylinderDetection(const std::string& yml_file_name,
                      const std::string& video_read_path,
                      const std::string& video_write_path) {
    Model model;
    //model.Load(yml_file_name);
    model.LoadBinary(yml_file_name);
    
    PnPProblem pnp_detection(param_IPhone7P, dist_coeff);
    PnPProblem pnp_detection_est(param_IPhone7P, dist_coeff);
    cv::Mat inliers_idx;

    cv::namedWindow("DETECTING", CV_WINDOW_AUTOSIZE);
    
    RobustMatcher rmatcher;
    cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(50);
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(num_keypoints, 1.2f, 4);
    rmatcher.SetFeatureDetector(detector);
    
    rmatcher.SetDescriptorMatcher(matcher);
    rmatcher.SetRatio(0.8f);
    
    //Kalman filter
    cv::KalmanFilter KF;         // instantiate Kalman Filter
    const int nStates = 18;            // the number of states
    const int nMeasurements = 6;       // the number of measured states
    const int nInputs = 0;             // the number of control actions
    const double dt = 0.125;           // time between measurements (1/FPS)
    
    InitKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
    cv::Mat measurements(nMeasurements, 1, CV_64F); measurements.setTo(cv::Scalar(0));
    bool good_measurement = false;

    //process video
    std::vector<cv::Point3f> list_points3d_model = model.GetPoints3d();
    cv::Mat descriptors_model = model.GetDescriptors();
    
    cv::VideoCapture cap;
    cap.open(video_read_path);
    if(!cap.isOpened()) {
        std::cout << "Could not open the camera device" << std::endl;
        return -1;
    }
    
    cv::VideoWriter outputVideo;                                        // Open the output
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    cv::Size S = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) );
    outputVideo.open(video_write_path , ex, cap.get(CV_CAP_PROP_FPS), S, true);
    
    cv::Mat frame_ori, frame_vis, frame;
    while(cap.read(frame_ori) && cv::waitKey(30) != 27) {
        frame_vis = frame_ori.clone();
        cv::undistort(frame_ori, frame, pnp_detection.Get_A_matrix(), pnp_detection.GetDistCoeffMatrix());
        
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::KeyPoint> keypoints_sence;
        //robust matching between model descriptors and sence descriptors
        rmatcher.RobustMatch(frame, good_matches, keypoints_sence, descriptors_model);
        
        //find out 2d/3d correspondences
        std::vector<cv::Point3f> list_points3d_model_match;
        std::vector<cv::Point2f> list_points2d_scene_match;
        
        for(unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            cv::Point3f point3d_model = list_points3d_model[good_matches[match_index].trainIdx];
            cv::Point2f point2d_scene = keypoints_sence[good_matches[match_index].queryIdx].pt;
            list_points3d_model_match.push_back(point3d_model);
            list_points2d_scene_match.push_back(point2d_scene);
        }
        
        Draw2DPoints(frame_vis, list_points2d_scene_match, cv::Scalar(0, 0, 255));
        
        std::vector<cv::Point2f> list_point2d_inliers;
        std::vector<cv::Point2f> list_point3d_inliers;
        if(good_matches.size() > 0) {
            pnp_detection.EstimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match,
                                             pnp_method, inliers_idx, iter_count, reprojection_error, confidence);
            
            // catch the inliers keypoints
            for(int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
                int n = inliers_idx.at<int>(inliers_index);
                cv::Point2f point2d = list_points2d_scene_match[n];
                list_point2d_inliers.push_back(point2d);
                
                cv::Point3f point3d = list_points3d_model_match[n];
                list_point3d_inliers.push_back(pnp_detection.Backproject3DPoint(point3d));
            }
            
            Draw2DPoints(frame_vis, list_point2d_inliers, cv::Scalar(255, 255, 255));
            
            // Kalman Filter
            
            good_measurement = false;
            if( inliers_idx.rows >= minInliersKalman) {
                // Get the measured translation
                cv::Mat translation_measured(3, 1, CV_64F);
                translation_measured = pnp_detection.Get_t_matrix();
                
                // Get the measured rotation
                cv::Mat rotation_measured(3, 3, CV_64F);
                rotation_measured = pnp_detection.Get_R_matrix();
                
                // fill the measurements vector
                FillMeasurements(measurements, translation_measured, rotation_measured);
                
                good_measurement = true;
            }
            // Instantiate estimated translation and rotation
            cv::Mat translation_estimated(3, 1, CV_64F);
            cv::Mat rotation_estimated(3, 3, CV_64F);
            
            // update the Kalman filter with good measurements
            UpdateKalmanFilter( KF, measurements,
                               translation_estimated, rotation_estimated);
            
            
            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.Set_P_matrix(rotation_estimated, translation_estimated);
        }
        
        good_measurement = true;
        //draw pose
        float l = 100;
        std::vector<cv::Point2f> pose_points2d;
        if(good_measurement) {
            pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0, 0, 0)));
            pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(l,0,0)));  // axis x
            pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0,l,0)));  // axis y
            pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0,0,l)));  // axis z
        } else {
            pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0, 0, 0)));
            pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(l,0,0)));  // axis x
            pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0,l,0)));  // axis y
            pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0,0,l)));  // axis z
        }
        Draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        
        cv::imshow("DETECTING", frame_vis);
        
        outputVideo << frame_vis;
    }
    
    return 0;
}



///////////////////////////////////////////

/**********************************************************************************************************/
void InitKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt) {
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2));   // set measurement noise
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance
    
    
    /** DYNAMIC MODEL **/
    
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    
    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
    
    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
    
    
    /** MEASUREMENT MODEL **/
    
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    
    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
    
}

/**********************************************************************************************************/
void UpdateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                        cv::Mat &translation_estimated,
                        cv::Mat &rotation_estimated ) {
    
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

/**********************************************************************************************************/
void FillMeasurements( cv::Mat &measurements,
                      const cv::Mat &translation_measured,
                      const cv::Mat &rotation_measured) {
    // Convert rotation matrix to euler angles
    cv::Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = rot2euler(rotation_measured);
    
    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

