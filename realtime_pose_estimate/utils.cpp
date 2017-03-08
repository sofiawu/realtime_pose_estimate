//
//  utils.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "utils.hpp"
#include <string>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace posest {
    
    // For text
    const int font_face = cv::FONT_ITALIC;
    const double font_scale = 0.75;
    const int thickness_font = 2;
    
    // For circles
    const int line_type = 8;
    const int radius = 4;
    const double thickness_circ = -1;
    
    // Draw a text with the question point
    void DrawQuestion(cv::Mat image, cv::Point3f point, cv::Scalar color) {
        std::string x = IntToString((int)point.x);
        std::string y = IntToString((int)point.y);
        std::string z = IntToString((int)point.z);
        
        std::string text = " Where is point (" + x + ","  + y + "," + z + ") ?";
        cv::putText(image, text, cv::Point(25,50), font_face, font_scale, color, thickness_font, 8);
    }
    // Draw a text with the number of entered points ( (25,50) (25, 75) )
    void DrawText(cv::Mat image, std::string text, cv::Scalar color, cv::Point point) {
        cv::putText(image, text, point, font_face, font_scale, color, thickness_font, 8);
    }
    // Draw a text with the frame ratio
    void DrawFPS(cv::Mat image, double fps, cv::Scalar color) {
        std::string fps_str = IntToString((int)fps);
        std::string text = fps_str + " FPS";
        cv::putText(image, text, cv::Point(500,50), font_face, font_scale, color, thickness_font, 8);
    }
    // Draw a text with the frame ratio
    void DrawConfidence(cv::Mat image, double confidence, cv::Scalar color) {
        std::string conf_str = IntToString((int)confidence);
        std::string text = conf_str + " %";
        cv::putText(image, text, cv::Point(500,75), font_face, font_scale, color, thickness_font, 8);
    }
    // Draw a text with the number of entered points
    void DrawCounter(cv::Mat image, int n, int n_max, cv::Scalar color) {
        std::string n_str = IntToString(n);
        std::string n_max_str = IntToString(n_max);
        std::string text = n_str + " of " + n_max_str + " points";
        cv::putText(image, text, cv::Point(500,50), font_face, font_scale, color, thickness_font, 8);
    }
    // Draw the points and the coordinates
    void DrawPoints(cv::Mat image,
                    std::vector<cv::Point2f> &list_points_2d,
                    std::vector<cv::Point3f> &list_points_3d,
                    cv::Scalar color) {
        for (unsigned int i = 0; i < list_points_2d.size(); ++i) {
            cv::Point2f point_2d = list_points_2d[i];
            cv::Point3f point_3d = list_points_3d[i];
            
            // Draw Selected points
            cv::circle(image, point_2d, radius, color, -1, line_type );
            
            std::string idx = IntToString(i+1);
            std::string x = IntToString((int)point_3d.x);
            std::string y = IntToString((int)point_3d.y);
            std::string z = IntToString((int)point_3d.z);
            std::string text = "P" + idx + " (" + x + "," + y + "," + z +")";
            
            point_2d.x = point_2d.x + 10;
            point_2d.y = point_2d.y - 10;
            cv::putText(image, text, point_2d, font_face, font_scale * 0.5, color, thickness_font, 8);
        }
    }
    // Draw only the 2D points
    void Draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color) {
        for( size_t i = 0; i < list_points.size(); i++) {
            cv::Point2f point_2d = list_points[i];
            
            // Draw Selected points
            cv::circle(image, point_2d, radius, color, -1, line_type );
        }
    }
    // Draw an arrow into the image
    void DrawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color,
                   int arrow_magnitude, int thickness, int line_type, int shift) {
        //Draw the principle line
        cv::line(image, p, q, color, thickness, line_type, shift);
        const double PI = CV_PI;
        //compute the angle alpha
        double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
        //compute the coordinates of the first segment
        p.x = (int) ( q.x +  arrow_magnitude * cos(angle + PI/4));
        p.y = (int) ( q.y +  arrow_magnitude * sin(angle + PI/4));
        //Draw the first segment
        cv::line(image, p, q, color, thickness, line_type, shift);
        //compute the coordinates of the second segment
        p.x = (int) ( q.x +  arrow_magnitude * cos(angle - PI/4));
        p.y = (int) ( q.y +  arrow_magnitude * sin(angle - PI/4));
        //Draw the second segment
        cv::line(image, p, q, color, thickness, line_type, shift);
    }
    // Draw the 3D coordinate axes
    void Draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d) {
        cv::Scalar red(0, 0, 255);
        cv::Scalar green(0, 255, 0);
        cv::Scalar blue(255, 0, 0);
        cv::Scalar black(0, 0, 0);
        
        cv::Point2i origin = list_points2d[0];
        cv::Point2i pointX = list_points2d[1];
        cv::Point2i pointY = list_points2d[2];
        cv::Point2i pointZ = list_points2d[3];
        
        DrawArrow(image, origin, pointX, red, 9, 2);
        DrawArrow(image, origin, pointY, blue, 9, 2);
        DrawArrow(image, origin, pointZ, green, 9, 2);
        cv::circle(image, origin, radius / 2, black, -1, line_type );
    }

    // Draw the object mesh
    void DrawObjectMesh(cv::Mat image, const Mesh *mesh,
                        PnPProblem *pnp_problem, cv::Scalar color) {
        std::vector<std::vector<int> > list_triangles = mesh->GetTrianglesList();
        for( size_t i = 0; i < list_triangles.size(); i++) {
            std::vector<int> tmp_triangle = list_triangles.at(i);
            
            cv::Point3f point_3d_0 = mesh->GetVertex(tmp_triangle[0]);
            cv::Point3f point_3d_1 = mesh->GetVertex(tmp_triangle[1]);
            cv::Point3f point_3d_2 = mesh->GetVertex(tmp_triangle[2]);
            
            cv::Point2f point_2d_0 = pnp_problem->Backproject3DPoint(point_3d_0);
            cv::Point2f point_2d_1 = pnp_problem->Backproject3DPoint(point_3d_1);
            cv::Point2f point_2d_2 = pnp_problem->Backproject3DPoint(point_3d_2);
            
            cv::line(image, point_2d_0, point_2d_1, color, 1);
            cv::line(image, point_2d_1, point_2d_2, color, 1);
            cv::line(image, point_2d_2, point_2d_0, color, 1);
        }
    }
    
    ////////////
    
    // Computes the norm of the translation error
    double get_translation_error(const cv::Mat &t_true, const cv::Mat &t) {
        return cv::norm( t_true - t );
    }
    
    // Computes the norm of the rotation error
    double get_rotation_error(const cv::Mat &R_true, const cv::Mat &R) {
        cv::Mat error_vec, error_mat;
        error_mat = R_true * cv::Mat(R.inv()).mul(-1);
        cv::Rodrigues(error_mat, error_vec);
        
        return cv::norm(error_vec);
    }
    
    // Converts a given Rotation Matrix to Euler angles
    cv::Mat rot2euler(const cv::Mat & rotationMatrix) {
        cv::Mat euler(3,1,CV_64F);
        
        double m00 = rotationMatrix.at<double>(0,0);
        double m02 = rotationMatrix.at<double>(0,2);
        double m10 = rotationMatrix.at<double>(1,0);
        double m11 = rotationMatrix.at<double>(1,1);
        double m12 = rotationMatrix.at<double>(1,2);
        double m20 = rotationMatrix.at<double>(2,0);
        double m22 = rotationMatrix.at<double>(2,2);
        
        double x, y, z;
        
        // Assuming the angles are in radians.
        if (m10 > 0.998) { // singularity at north pole
            x = 0;
            y = CV_PI / 2;
            z = atan2(m02,m22);
        }
        else if (m10 < -0.998) { // singularity at south pole
            x = 0;
            y = -CV_PI / 2;
            z = atan2(m02,m22);
        } else {
            x = atan2(-m12,m11);
            y = asin(m10);
            z = atan2(-m20,m00);
        }
        
        euler.at<double>(0) = x;
        euler.at<double>(1) = y;
        euler.at<double>(2) = z;
        
        return euler;
    }
    
    // Converts a given Euler angles to Rotation Matrix
    cv::Mat euler2rot(const cv::Mat & euler) {
        cv::Mat rotationMatrix(3, 3, CV_64F);
        
        double x = euler.at<double>(0);
        double y = euler.at<double>(1);
        double z = euler.at<double>(2);
        
        // Assuming the angles are in radians.
        double ch = cos(z);
        double sh = sin(z);
        double ca = cos(y);
        double sa = sin(y);
        double cb = cos(x);
        double sb = sin(x);
        
        double m00, m01, m02, m10, m11, m12, m20, m21, m22;
        
        m00 = ch * ca;
        m01 = sh * sb - ch * sa * cb;
        m02 = ch * sa * sb + sh * cb;
        m10 = sa;
        m11 = ca * cb;
        m12 = -ca * sb;
        m20 = -sh * ca;
        m21 = sh * sa * cb + ch * sb;
        m22 = -sh * sa * sb + ch * cb;
        
        rotationMatrix.at<double>(0,0) = m00;
        rotationMatrix.at<double>(0,1) = m01;
        rotationMatrix.at<double>(0,2) = m02;
        rotationMatrix.at<double>(1,0) = m10;
        rotationMatrix.at<double>(1,1) = m11;
        rotationMatrix.at<double>(1,2) = m12;
        rotationMatrix.at<double>(2,0) = m20;
        rotationMatrix.at<double>(2,1) = m21;
        rotationMatrix.at<double>(2,2) = m22;
        
        return rotationMatrix;
    }

    ////////////
    
    // Converts a given string to an integer
    int StringToInt ( const std::string &text ) {
        std::istringstream ss(text);
        int result;
        return ss >> result ? result : 0;
    }
    
     // Converts a given string to a float
    float StringToFloat(const std::string& text) {
        std::istringstream ss(text);
        float result;
        return ss >> result ? result : 0.0f;
    }
    
    // Converts a given float to a string
    std::string FloatToString ( float num ) {
        std::ostringstream ss;
        ss << num;
        return ss.str();
    }
    
    // Converts a given integer to a string
    std::string IntToString ( int num ) {
        std::ostringstream ss;
        ss << num;
        return ss.str();
    }
    
}
