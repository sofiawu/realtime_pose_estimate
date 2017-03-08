//
//  utils.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <string>

#include "pnp_problem.hpp"

namespace posest {
    
    // Draw a text with the question point
    void DrawQuestion(cv::Mat image, cv::Point3f point, cv::Scalar color);
    
    // Draw a text with the number of entered points
    void DrawText(cv::Mat image, std::string text, cv::Scalar color, cv::Point point);
    
    // Draw a text with the frame ratio
    void DrawFPS(cv::Mat image, double fps, cv::Scalar color);
    // Draw a text with the frame ratio
    void DrawConfidence(cv::Mat image, double confidence, cv::Scalar color);
    
    // Draw a text with the number of entered points
    void DrawCounter(cv::Mat image, int n, int n_max, cv::Scalar color);
    
    // Draw the points and the coordinates
    void DrawPoints(cv::Mat image, std::vector<cv::Point2f> &list_points_2d, std::vector<cv::Point3f> &list_points_3d, cv::Scalar color);
    
    // Draw only the 2D points
    void Draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);
    
    // Draw an arrow into the image
    void DrawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);
    
    // Draw the 3D coordinate axes
    void Draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d);
    
    // Draw the object mesh
    void DrawObjectMesh(cv::Mat image, const Mesh *mesh, PnPProblem *pnpProblem, cv::Scalar color);

    ////////
    
    // Computes the norm of the translation error
    double get_translation_error(const cv::Mat &t_true, const cv::Mat &t);
    
    // Computes the norm of the rotation error
    double get_rotation_error(const cv::Mat &R_true, const cv::Mat &R);
    
    // Converts a given Rotation Matrix to Euler angles
    cv::Mat rot2euler(const cv::Mat & rotationMatrix);
    
    // Converts a given Euler angles to Rotation Matrix
    cv::Mat euler2rot(const cv::Mat & euler);
    
    ///////
    
    // Converts a given string to an integer
    int StringToInt ( const std::string &text );
    
    // Converts a given string to a float
    float StringToFloat(const std::string& text);
    
    // Converts a given float to a string
    std::string FloatToString ( float num );
    
    // Converts a given integer to a string
    std::string IntToString ( int num );
}

#endif /* utils_hpp */