//
//  model_registration.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "model_registration.hpp"

namespace posest {
    ModelRegistration::ModelRegistration() {
        n_registrations_ = 0;
        max_registrations_ = 0;
    }
    
    ModelRegistration::~ModelRegistration() {
        // TODO Auto-generated destructor stub
    }
    
    void ModelRegistration::RegisterPoint(const cv::Point2f &point2d, const cv::Point3f &point3d) {
        // add correspondence at the end of the vector
        list_points2d_.push_back(point2d);
        list_points3d_.push_back(point3d);
        n_registrations_++;
    }
    
    void ModelRegistration::Reset() {
        n_registrations_ = 0;
        max_registrations_ = 0;
        list_points2d_.clear();
        list_points3d_.clear();
    }

}
