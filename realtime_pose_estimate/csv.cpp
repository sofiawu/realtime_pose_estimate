//
//  csv_reader.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "csv.hpp"
#include "utils.hpp"

namespace posest {
    
    /** The default constructor of the CSV reader Class */
    CsvReader::CsvReader(const std::string &path, const char &separator){
        file_.open(path.c_str(), std::ifstream::in);
        _separator = separator;
    }
    
    /* Read a plane text file with .ply format */
    void CsvReader::ReadPLY(std::vector<cv::Point3f> &list_vertex, std::vector<std::vector<int> > &list_triangles) {
        std::string line, tmp_str, n;
        int num_vertex = 0, num_triangles = 0;
        int count = 0;
        bool end_header = false;
        bool end_vertex = false;
        
        // Read the whole *.ply file
        while (getline(file_, line)) {
            std::stringstream liness(line);
            
            // read header
            if(!end_header) {
                getline(liness, tmp_str, _separator);
                if( tmp_str == "element" ) {
                    getline(liness, tmp_str, _separator);
                    getline(liness, n);
                    if(tmp_str == "vertex") num_vertex = StringToInt(n);
                    if(tmp_str == "face") num_triangles = StringToInt(n);
                }
                if(tmp_str == "end_header") end_header = true;
            } else if(end_header) { // read file content
                // read vertex and add into 'list_vertex'
                if(!end_vertex && count < num_vertex) {
                    std::string x, y, z;
                    getline(liness, x, _separator);
                    getline(liness, y, _separator);
                    getline(liness, z);
                    
                    cv::Point3f tmp_p;
                    tmp_p.x = StringToFloat(x);
                    tmp_p.y = StringToFloat(y);
                    tmp_p.z = StringToFloat(z);
                    list_vertex.push_back(tmp_p);
                    
                    count++;
                    if(count == num_vertex) {
                        count = 0;
                        end_vertex = !end_vertex;
                    }
                } else if(end_vertex  && count < num_triangles) { // read faces and add into 'list_triangles'
                    std::string num_pts_per_face, id0, id1, id2;
                    getline(liness, num_pts_per_face, _separator);
                    getline(liness, id0, _separator);
                    getline(liness, id1, _separator);
                    getline(liness, id2);
                    
                    std::vector<int> tmp_triangle(3);
                    tmp_triangle[0] = StringToInt(id0);
                    tmp_triangle[1] = StringToInt(id1);
                    tmp_triangle[2] = StringToInt(id2);
                    list_triangles.push_back(tmp_triangle);
                    
                    count++;
                }
            }
        }
    }

    ////////////////////////
    
    CsvWriter::CsvWriter(const std::string &path, const std::string &separator){
        file_.open(path.c_str(), std::ofstream::out);
        is_first_term_ = true;
        separator_ = separator;
    }
    
    CsvWriter::~CsvWriter() {
        file_.flush();
        file_.close();
    }
    
    void CsvWriter::WriteXYZ(const std::vector<cv::Point3f> &list_points3d) {
        std::string x, y, z;
        for(unsigned int i = 0; i < list_points3d.size(); ++i) {
            x = FloatToString(list_points3d[i].x);
            y = FloatToString(list_points3d[i].y);
            z = FloatToString(list_points3d[i].z);
            
            file_ << x << separator_ << y << separator_ << z << std::endl;
        }
        
    }
    
    void CsvWriter::WriteUVXYZ(const std::vector<cv::Point3f> &list_points3d, const std::vector<cv::Point2f> &list_points2d, const cv::Mat &descriptors) {
        std::string u, v, x, y, z, descriptor_str;
        for(unsigned int i = 0; i < list_points3d.size(); ++i) {
            u = FloatToString(list_points2d[i].x);
            v = FloatToString(list_points2d[i].y);
            x = FloatToString(list_points3d[i].x);
            y = FloatToString(list_points3d[i].y);
            z = FloatToString(list_points3d[i].z);
            
            file_ << u << separator_ << v << separator_ << x << separator_ << y << separator_ << z;
            
            for(int j = 0; j < 32; ++j) { //256 bit orb
                descriptor_str = FloatToString(descriptors.at<float>(i,j));
                file_ << separator_ << descriptor_str;
            }
            file_ << std::endl;
        }
    }

}
