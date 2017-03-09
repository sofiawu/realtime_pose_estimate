//
//  cylinder_registration.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "estimate.h"
#include <string>

#include "robust_matcher.hpp"
#include "model.hpp"

using namespace posest;

static const int num_keypoints = 3000;

int CylinderRegister(const std::string& side_file,
                     const std::string& write_path) {
    cv::Mat img_in = cv::imread(side_file, 0);
    if(img_in.empty()) {
        printf("%s is not exist.", side_file.c_str());
        return -1;
    }
    cv::Mat img_out;
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    RobustMatcher rmatcher;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(num_keypoints, 1.2f, 10);
    rmatcher.SetFeatureDetector(detector);
    rmatcher.ComputeKeyPoints(img_in, keypoints);
    rmatcher.ComputeDescriptors(img_in, keypoints, descriptors);
    
    cv::namedWindow("REGISTRATION", CV_WINDOW_AUTOSIZE);
    
    cv::drawKeypoints(img_in, keypoints, img_out);
    cv::imshow("REGISTRATION", img_out);
    cv::waitKey(0);
    
    //compute radius
    Model model;
    
    double side_len = (double)img_in.cols;
    double radius = side_len / (2.0 * CV_PI);
    
    std::vector<cv::Point3f> list_points3d;
    for(unsigned int i = 0; i < keypoints.size(); i++) {
        cv::Point2f point2d(keypoints[i].pt);
        cv::Point3f point3d;
        point3d.z = point2d.y;
        
        double angle = point2d.x / radius;
        point3d.y = radius * cos(angle);
        point3d.x = radius * sin(angle);
        
        model.AddCorrespondence(point2d, point3d);
        model.AddDescriptor(descriptors.row(i));
        model.AddKeypoint(keypoints[i]);
    }
    model.SetModelType(2);
    model.SetModelParam(448.8, 448.8, 404);
    //model.Save(write_path);
    model.SaveBinary(write_path);
    return 0;
}
