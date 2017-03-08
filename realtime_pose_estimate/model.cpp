//
//  model.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "model.hpp"

namespace posest {
    
    Model::Model() : list_points2d_in_(0), list_points2d_out_(0), list_points3d_in_(0), n_correspondences_(0) {}
    
    Model::~Model() {
        // TODO Auto-generated destructor stub
    }
    
    void Model::AddCorrespondence(const cv::Point2f &point2d, const cv::Point3f &point3d) {
        list_points2d_in_.push_back(point2d);
        list_points3d_in_.push_back(point3d);
        n_correspondences_++;
    }
    
    void Model::AddOutlier(const cv::Point2f &point2d) {
        list_points2d_out_.push_back(point2d);
    }
    
    void Model::AddDescriptor(const cv::Mat &descriptor) {
        descriptors_.push_back(descriptor);
    }
    
    void Model::AddKeypoint(const cv::KeyPoint &kp) {
        list_keypoints_.push_back(kp);
    }
    
    
    /** Save a CSV file and fill the object mesh */
    void Model::Save(const std::string path) {
        cv::Mat points3dmatrix = cv::Mat(list_points3d_in_);
        cv::Mat points2dmatrix = cv::Mat(list_points2d_in_);
        
        cv::FileStorage storage(path, cv::FileStorage::WRITE);
        storage << "points_3d" << points3dmatrix;
        storage << "points_2d" << points2dmatrix;
        storage << "keypoints" << list_keypoints_;
        storage << "descriptors" << descriptors_;
        
        storage.release();
    }
    
    /** Load a YAML file using OpenCv functions **/
    void Model::Load(const std::string path) {
        cv::Mat points3d_mat;
        
        cv::FileStorage storage(path, cv::FileStorage::READ);
        storage["points_3d"] >> points3d_mat;
        storage["descriptors"] >> descriptors_;
        
        points3d_mat.copyTo(list_points3d_in_);
        
        storage.release();
    }
    
    void Model::Merge(const std::string& path) {
        cv::Mat points3d_mat;
        cv::Mat desc;
        std::vector<cv::Point3f> list_points3d;
        
        cv::FileStorage storage(path, cv::FileStorage::READ);
        storage["points_3d"] >> points3d_mat;
        storage["descriptors"] >> desc;
        
        points3d_mat.copyTo(list_points3d);
        
        for(int i = 0; i < list_points3d.size(); i++) {
            list_points3d_in_.push_back(list_points3d[i]);
        }
        
        cv::Mat temp(descriptors_.rows + desc.rows, desc.cols, desc.type());
        if(descriptors_.rows > 0) {
            cv::Mat temp_top = temp.rowRange(0, descriptors_.rows);
            descriptors_.copyTo(temp_top);
        }
        cv::Mat temp_bottom = temp.rowRange(descriptors_.rows, descriptors_.rows + desc.rows);
        desc.copyTo(temp_bottom);
        
        descriptors_.release();
        descriptors_ = temp.clone();
        
        storage.release();
    }
    
    void Model::MergeSave(const std::string& path) {
        cv::Mat points3dmatrix = cv::Mat(list_points3d_in_);
        
        cv::FileStorage storage(path, cv::FileStorage::WRITE);
        storage << "points_3d" << points3dmatrix;
        storage << "descriptors" << descriptors_;
        
        storage.release();
    }

}
