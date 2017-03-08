//
//  csv_reader.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef csv_reader_hpp
#define csv_reader_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>

namespace posest {
    class CsvReader {
    public:
        /**
         * The default constructor of the CSV reader Class.
         * The default separator is ' ' (empty space)
         *
         * @param path - The path of the file to read
         * @param separator - The separator character between words per line
         */
        CsvReader(const std::string &path, const char &separator = ' ');
        
        /**
         * Read a plane text file with .ply format
         *
         * @param list_vertex - The container of the vertices list of the mesh
         * @param list_triangles - The container of the triangles list of the mesh
         */
        void ReadPLY(std::vector<cv::Point3f> &list_vertex, std::vector<std::vector<int> > &list_triangles);
        
    private:
        /** The current stream file for the reader */
        std::ifstream file_;
        /** The separator character between words for each line */
        char _separator;
    };
    
    
    /////////
    class CsvWriter {
    public:
        CsvWriter(const std::string &path, const std::string &separator = " ");
        ~CsvWriter();
        void WriteXYZ(const std::vector<cv::Point3f> &list_points3d);
        void WriteUVXYZ(const std::vector<cv::Point3f> &list_points3d,
                        const std::vector<cv::Point2f> &list_points2d,
                        const cv::Mat &descriptors);
        
    private:
        std::ofstream file_;
        std::string separator_;
        bool is_first_term_;
    };

}

#endif /* csv_reader_hpp */
