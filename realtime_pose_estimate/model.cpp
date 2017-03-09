//
//  model.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "model.hpp"

namespace posest {
    
    Model::Model() : list_points2d_in_(0), list_points2d_out_(0), list_points3d_in_(0), n_correspondences_(0), model_type_(0) {}
    
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
    void Model::Save(const std::string& path) {
        cv::Mat points3dmatrix = cv::Mat(list_points3d_in_);
        cv::Mat points2dmatrix = cv::Mat(list_points2d_in_);
        
        cv::FileStorage storage(path, cv::FileStorage::WRITE);
        storage << "points_3d" << points3dmatrix;
        storage << "points_2d" << points2dmatrix;
        storage << "keypoints" << list_keypoints_;
        storage << "descriptors" << descriptors_;
        
        storage.release();
    }
    
    bool Model::SaveBinary(const std::string& path) {
        FILE* fp = fopen(path.c_str(), "wb");
        
        fwrite(&model_type_, sizeof(int), 1, fp);
        
        if(model_type_ == 1) {
            fwrite(&model_param_.cuboid.width, sizeof(float), 1, fp);
            fwrite(&model_param_.cuboid.height, sizeof(float), 1, fp);
            fwrite(&model_param_.cuboid.length, sizeof(float), 1, fp);
        } else if(model_type_ == 2) {
            fwrite(&model_param_.cylinder.bottom_diameter, sizeof(float), 1, fp);
            fwrite(&model_param_.cylinder.top_diameter, sizeof(float), 1, fp);
            fwrite(&model_param_.cylinder.side_length, sizeof(float), 1, fp);
        }
        
        //save 3d points
        int num_points = (int)list_points3d_in_.size();
        fwrite(&num_points, sizeof(int), 1, fp);
        for(auto point : list_points3d_in_) {
            fwrite(&point.x, sizeof(float), 1, fp);
            fwrite(&point.y, sizeof(float), 1, fp);
            fwrite(&point.z, sizeof(float), 1, fp);
        }
        //save descriptors
        int rows = (int)descriptors_.rows;
        int cols = (int)descriptors_.cols;
        int type = (int)descriptors_.type();
        fwrite(&rows, sizeof(int), 1, fp);
        fwrite(&cols, sizeof(int), 1, fp);
        fwrite(&type, sizeof(int), 1, fp);
        
        fwrite(descriptors_.data, sizeof(char), descriptors_.step * rows, fp);
        fclose(fp);
        return true;
    }
    
    /** Load a YAML file using OpenCv functions **/
    void Model::Load(const std::string& path) {
        cv::Mat points3d_mat;
        
        cv::FileStorage storage(path, cv::FileStorage::READ);
        storage["points_3d"] >> points3d_mat;
        storage["descriptors"] >> descriptors_;
        
        points3d_mat.copyTo(list_points3d_in_);
        
        storage.release();
    }
    
    bool Model::LoadBinary(const std::string &path) {
        FILE* fp = fopen(path.c_str(), "rb");
        if(fp == nullptr) {
            printf("%s is not exist.", path.c_str());
            return false;
        }
        
        fread(&model_type_, sizeof(int), 1, fp);
        
        if(model_type_ == 1) {
            fread(&model_param_.cuboid.width, sizeof(float), 1, fp);
            fread(&model_param_.cuboid.height, sizeof(float), 1, fp);
            fread(&model_param_.cuboid.length, sizeof(float), 1, fp);
        } else if(model_type_ == 2) {
            fread(&model_param_.cylinder.bottom_diameter, sizeof(float), 1, fp);
            fread(&model_param_.cylinder.top_diameter, sizeof(float), 1, fp);
            fread(&model_param_.cylinder.side_length, sizeof(float), 1, fp);
        }
        
        //load points
        int num_points;
        fread(&num_points, sizeof(int), 1, fp);
        list_points3d_in_.resize(num_points);
        for(int i = 0; i < num_points; ++i) {
            fread(&list_points3d_in_[i].x, sizeof(float), 1, fp);
            fread(&list_points3d_in_[i].y, sizeof(float), 1, fp);
            fread(&list_points3d_in_[i].z, sizeof(float), 1, fp);
        }
        //load descriptors
        int rows, cols, type;
        fread(&rows, sizeof(int), 1, fp);
        fread(&cols, sizeof(int), 1, fp);
        fread(&type, sizeof(int), 1, fp);
        
        descriptors_ = cv::Mat(rows, cols, type);
        fread(descriptors_.data, sizeof(char), descriptors_.step * rows, fp);
        
        fclose(fp);
        
        return true;
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
