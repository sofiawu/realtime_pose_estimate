//
//  cube_detection.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include <stdio.h>
#include <iostream>

#include "pnp_problem.hpp"
#include "mesh.hpp"
#include "model.hpp"
#include "robust_matcher.hpp"
#include "model_registration.hpp"

#include "utils.hpp"

#include <opencv2//core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include "estimate.h"

using namespace posest;

///////

static double width = 1000, height = 780;
static double params_WEBCAM[] = { 0.8638 * width,   // fx
    1.1527 * height,  // fy
    0.5012 * width - 0.5,      // cx
    0.4975 * height - 0.5};    // cy
// Some basic colors
cv::Scalar red(0, 0, 255);
cv::Scalar green(0,255,0);
cv::Scalar blue(255,0,0);
cv::Scalar yellow(0,255,255);

// Robust Matcher parameters
static const int num_keypoints = 800;      // number of detected keypoints
static const float ratio_test = 0.90f;          // ratio test
static bool fast_match = false;       // fastRobustMatch() or robustMatch()

// RANSAC parameters
static const int iterationsCount = 500;      // number of Ransac iterations.
static const float reprojectionError = 3.0;  // maximum allowed distance to consider it an inlier.
static const double confidence = 0.95;        // ransac successful confidence.

// Kalman Filter parameters
static const int minInliersKalman = 30;    // Kalman threshold updating

// PnP parameters
static const int pnpMethod = cv::SOLVEPNP_ITERATIVE;

static void InitKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);
static void UpdateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                        cv::Mat &translation_estimated, cv::Mat &rotation_estimated );
static void FillMeasurements( cv::Mat &measurements,
                      const cv::Mat &translation_measured, const cv::Mat &rotation_measured);

int CubeDetection(const std::string& yml_read_path,
                  const std::string& ply_read_path,
                  const std::string& video_read_path,
                  const std::string& video_write_path) {
    PnPProblem pnp_detection(params_WEBCAM);
    PnPProblem pnp_detection_est(params_WEBCAM);
    
    Model model;               // instantiate Model object
    model.Load(yml_read_path); // load a 3D textured object model
    
    Mesh mesh;                 // instantiate Mesh object
    mesh.Load(ply_read_path);  // load an object mesh
    
    RobustMatcher rmatcher;                                                     // instantiate RobustMatcher
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(num_keypoints);
    rmatcher.SetFeatureDetector(orb);                                      // set feature detector
    rmatcher.SetDescriptorExtractor(orb);                                 // set descriptor extractor
    
    cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1); // instantiate LSH index parameters
    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(50);       // instantiate flann search parameters
    
    // instantiate FlannBased matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
    rmatcher.SetDescriptorMatcher(matcher);                                                         // set matcher
    rmatcher.SetRatio(ratio_test); // set ratio test parameter

    cv::KalmanFilter KF;         // instantiate Kalman Filter
    const int nStates = 18;            // the number of states
    const int nMeasurements = 6;       // the number of measured states
    const int nInputs = 0;             // the number of control actions
    const double dt = 0.125;           // time between measurements (1/FPS)
    
    InitKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
    cv::Mat measurements(nMeasurements, 1, CV_64F); measurements.setTo(cv::Scalar(0));
    bool good_measurement = false;
    
    // Get the MODEL INFO
    std::vector<cv::Point3f> list_points3d_model = model.GetPoints3d();  // list with model 3D coordinates
    cv::Mat descriptors_model = model.GetDescriptors();                  // list with descriptors of each 3D coordinate
    
    // Create & Open Window
    cv::namedWindow("REAL TIME DEMO", cv::WINDOW_KEEPRATIO);
    
    cv::VideoCapture cap;                           // instantiate VideoCapture
    cap.open(video_read_path);                      // open a recorded video
    
    if(!cap.isOpened()) {   // check if we succeeded
        std::cout << "Could not open the camera device" << std::endl;
        return -1;
    }
    
    // start and end times
    time_t start, end;
    
    // fps calculated using number of frames / seconds
    // floating point seconds elapsed since start
    double fps, sec;
    
    // frame counter
    int counter = 0;
    
    // start the clock
    time(&start);
    
    cv::Mat frame, frame_vis;
    
    cv::VideoWriter outputVideo;                                        // Open the output
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    cv::Size S = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
                  (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) );
    outputVideo.open(video_write_path , ex, cap.get(CV_CAP_PROP_FPS), S, true);
    
    while(cap.read(frame) && cv::waitKey(30) != 27) { // capture frame until ESC is pressed
        frame_vis = frame.clone();    // refresh visualisation frame
        
        // -- Step 1: Robust matching between model descriptors and scene descriptors
        
        std::vector<cv::DMatch> good_matches;       // to obtain the 3D points of the model
        std::vector<cv::KeyPoint> keypoints_scene;  // to obtain the 2D points of the scene
        
        if(fast_match)
            rmatcher.FastRobustMatch(frame, good_matches, keypoints_scene, descriptors_model);
        else
            rmatcher.RobustMatch(frame, good_matches, keypoints_scene, descriptors_model);
        
        // -- Step 2: Find out the 2D/3D correspondences
        
        std::vector<cv::Point3f> list_points3d_model_match; // container for the model 3D coordinates found in the scene
        std::vector<cv::Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene
        
        for(unsigned int match_index = 0; match_index < good_matches.size(); ++match_index) {
            cv::Point3f point3d_model = list_points3d_model[ good_matches[match_index].trainIdx ];  // 3D point from model
            cv::Point2f point2d_scene = keypoints_scene[ good_matches[match_index].queryIdx ].pt; // 2D point from the scene
            list_points3d_model_match.push_back(point3d_model);         // add 3D point
            list_points2d_scene_match.push_back(point2d_scene);         // add 2D point
        }
        
        // Draw outliers
        Draw2DPoints(frame_vis, list_points2d_scene_match, red);
        
        cv::Mat inliers_idx;
        std::vector<cv::Point2f> list_points2d_inliers;
        
        if(good_matches.size() > 0) { // None matches, then RANSAC crashes
            // -- Step 3: Estimate the pose using RANSAC approach
            pnp_detection.EstimatePoseRANSAC( list_points3d_model_match, list_points2d_scene_match,
                                             pnpMethod, inliers_idx,
                                             iterationsCount, reprojectionError, confidence );
            
            // -- Step 4: Catch the inliers keypoints to draw
            for(int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
                int n = inliers_idx.at<int>(inliers_index);         // i-inlier
                cv::Point2f point2d = list_points2d_scene_match[n]; // i-inlier point 2D
                list_points2d_inliers.push_back(point2d);           // add i-inlier to list
            }
            
            // Draw inliers points 2D
            Draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 255, 0));
            
            // -- Step 5: Kalman Filter
            
            good_measurement = false;
            
            // GOOD MEASUREMENT
            if( inliers_idx.rows >= minInliersKalman ) {
            
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
        
        // -- Step X: Draw pose
        
        if(good_measurement)
            DrawObjectMesh(frame_vis, &mesh, &pnp_detection, green);  // draw current pose
        else
            DrawObjectMesh(frame_vis, &mesh, &pnp_detection_est, green); // draw estimated pose
        
        const float l = 5;
        std::vector<cv::Point2f> pose_points2d;
        pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0,0,0)));  // axis center
        pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(l,0,0)));  // axis x
        pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0,l,0)));  // axis y
        pose_points2d.push_back(pnp_detection_est.Backproject3DPoint(cv::Point3f(0,0,l)));  // axis z
        Draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        
        // FRAME RATE
        
        // see how much time has elapsed
        time(&end);
        
        // calculate current FPS
        ++counter;
        sec = difftime (end, start);
        
        fps = counter / sec;
        
        DrawFPS(frame_vis, fps, yellow); // frame ratio
        double detection_ratio = ((double)inliers_idx.rows/(double)good_matches.size())*100;
        DrawConfidence(frame_vis, detection_ratio, yellow);
        
        // -- Step X: Draw some debugging text
        
        // Draw some debug text
        int inliers_int = inliers_idx.rows;
        int outliers_int = (int)good_matches.size() - inliers_int;
        std::string inliers_str = IntToString(inliers_int);
        std::string outliers_str = IntToString(outliers_int);
        std::string n = IntToString((int)good_matches.size());
        std::string text = "Found " + inliers_str + " of " + n + " matches";
        std::string text2 = "Inliers: " + inliers_str + " - Outliers: " + outliers_str;
        
        DrawText(frame_vis, text, green, cv::Point(25, 50));
        DrawText(frame_vis, text2, red, cv::Point(25, 75));
        
        imshow("REAL TIME DEMO", frame_vis);
        
        outputVideo << frame_vis;
    }
    
    // Close and Destroy Window
    cv::destroyWindow("REAL TIME DEMO");
    
    std::cout << "GOODBYE . " << std::endl;
    return 0;
}

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
