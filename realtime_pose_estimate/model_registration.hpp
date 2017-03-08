//
//  model_registration.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef model_registration_hpp
#define model_registration_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>

namespace posest {
    
    class ModelRegistration {
    public:
        ModelRegistration();
        virtual ~ModelRegistration();
        
        inline void SetNumMax(int n) { max_registrations_ = n; }
        
        inline std::vector<cv::Point2f> GetPoints2d() const { return list_points2d_; }
        inline std::vector<cv::Point3f> GetPoints3d() const { return list_points3d_; }
        inline int GetNumMax() const { return max_registrations_; }
        inline int GetNumRegist() const { return n_registrations_; }
        
        inline bool IsRegistrable() const { return (n_registrations_ < max_registrations_); }
        void RegisterPoint(const cv::Point2f &point2d, const cv::Point3f &point3d);
        void Reset();
        
    private:
        /** The current number of registered points */
        int n_registrations_;
        /** The total number of points to register */
        int max_registrations_;
        /** The list of 2D points to register the model */
        std::vector<cv::Point2f> list_points2d_;
        /** The list of 3D points to register the model */
        std::vector<cv::Point3f> list_points3d_;
    };

}

#endif /* model_registration_hpp */
