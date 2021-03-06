//
//  model.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef model_hpp
#define model_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>

namespace posest {
    
    union ModelParam {
        struct {
            float length;
            float width;
            float height;
        } cuboid;
        struct {
            float bottom_diameter;
            float top_diameter;
            float side_length;
        } cylinder;
    };
    
    class Model {
    public:
        Model();
        virtual ~Model();
        
        inline std::vector<cv::Point2f> GetPoints2dIn() const { return list_points2d_in_; }
        inline std::vector<cv::Point2f> GetPoints2dOut() const { return list_points2d_out_; }
        inline std::vector<cv::Point3f> GetPoints3d() const { return list_points3d_in_; }
        inline std::vector<cv::KeyPoint> GetKeypoints() const { return list_keypoints_; }
        inline cv::Mat GetDescriptors() const { return descriptors_; }
        inline int GetNumDescriptors() const { return descriptors_.rows; }
        inline int GetModelType() const { return model_type_;}
        inline void SetModelType(int type) { model_type_ = type; }
        
        inline void SetModelParam(float a, float b, float c) {
            if(model_type_ == 1) {
                model_param_.cuboid.width = a;
                model_param_.cuboid.height = b;
                model_param_.cuboid.length = c;
            } else if(model_type_ == 2) {
                model_param_.cylinder.bottom_diameter = a;
                model_param_.cylinder.top_diameter = b;
                model_param_.cylinder.side_length = c;
            }
        }
        inline ModelParam GetModelParam() const { return model_param_; }
        
        void AddCorrespondence(const cv::Point2f &point2d, const cv::Point3f &point3d);
        void AddOutlier(const cv::Point2f &point2d);
        void AddDescriptor(const cv::Mat &descriptor);
        void AddKeypoint(const cv::KeyPoint &kp);
        
        void Save(const std::string& path);
        bool SaveBinary(const std::string& path);
        void Load(const std::string& path);
        bool LoadBinary(const std::string& path);
        void Merge(const std::string& path);
        void MergeSave(const std::string& path);
        
    private:
        /** The current number of correspondecnes */
        int n_correspondences_;
        /** The list of 2D points on the model surface */
        std::vector<cv::KeyPoint> list_keypoints_;
        /** The list of 2D points on the model surface */
        std::vector<cv::Point2f> list_points2d_in_;
        /** The list of 2D points outside the model surface */
        std::vector<cv::Point2f> list_points2d_out_;
        /** The list of 3D points on the model surface */
        std::vector<cv::Point3f> list_points3d_in_;
        /** The list of 2D points descriptors */
        cv::Mat descriptors_;
        
        int model_type_; // (0 : 2d image; 1: cube ; 2: cylinder; 3: general 3d)
        ModelParam model_param_;
    };
}

#endif /* model_hpp */
