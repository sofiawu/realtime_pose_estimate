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

using namespace posest;

const int num_keypoints = 800;

static const double width = 1920, height = 1080;
static const double params_WEBCAM[] = { 0.8638 * width,   // fx
    1.1527 * height,  // fy
    0.5012 * width - 0.5,      // cx
    0.4975 * height - 0.5};    // cy

static const int pnp_method = CV_ITERATIVE;
static const int iter_count = 500;
static const float reprojection_error = 3.0f;
static const double confidence = 0.95;


/////////

int CylinderDetection(const std::string& yml_file_name,
                      const std::string& video_read_path,
                      const std::string& video_write_path) {
    Model model;
    //model.Load(yml_file_name);
    model.LoadBinary(yml_file_name);
    
    PnPProblem pnp_detection(params_WEBCAM);
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
    
    cv::Mat frame, frame_vis;
    while(cap.read(frame) && cv::waitKey(30) != 27) {
        frame_vis = frame.clone();
        
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
            cv::imshow("DETECTING", frame_vis);
        }
        //draw pose
        float l = 100;
        std::vector<cv::Point2f> pose_points2d;
        pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0, 0, 0)));
        pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(l,0,0)));  // axis x
        pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0,l,0)));  // axis y
        pose_points2d.push_back(pnp_detection.Backproject3DPoint(cv::Point3f(0,0,l)));  // axis z
        Draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        cv::imshow("DETECTING", frame_vis);
        
        outputVideo << frame_vis;
    }
    
    return 0;
}
