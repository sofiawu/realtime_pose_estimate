//
//  pnp_problem.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef pnp_problem_hpp
#define pnp_problem_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include "mesh.hpp"

namespace posest {
    
    class PnPProblem {
    public:
        explicit PnPProblem(const double param[]);  // custom constructor
        explicit PnPProblem(const double param[], const double dist_coeff[]);
        virtual ~PnPProblem();
        
        bool Backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d);
        cv::Point2f Backproject3DPoint(const cv::Point3f &point3d);
        
        bool EstimatePose(const std::vector<cv::Point3f> &list_points3d,
                          const std::vector<cv::Point2f> &list_points2d,
                          int flags);
        void EstimatePoseRANSAC(const std::vector<cv::Point3f> &list_points3d,
                                const std::vector<cv::Point2f> &list_points2d,
                                int flags, cv::Mat &inliers,
                                int iterationsCount, float reprojectionError, double confidence );
        
        inline cv::Mat Get_A_matrix() const { return A_matrix_; }
        inline cv::Mat Get_R_matrix() const { return R_matrix_; }
        inline cv::Mat Get_t_matrix() const { return t_matrix_; }
        inline cv::Mat Get_P_matrix() const { return P_matrix_; }
        inline cv::Mat GetDistCoeffMatrix() const { return dist_coeffs_; }
        
        void Set_P_matrix( const cv::Mat &R_matrix, const cv::Mat &t_matrix);
        
        std::vector<cv::Point2f> VerifyPoints(Mesh *mesh);
        
    private:
        
        bool IntersectMollerTrumbore(Ray &R, Triangle &T, double *out);
        
    private:
        /** The calibration matrix */
        cv::Mat A_matrix_;
        /** The computed rotation matrix */
        cv::Mat R_matrix_;
        /** The computed translation matrix */
        cv::Mat t_matrix_;
        /** The computed projection matrix */
        cv::Mat P_matrix_;
        
        cv::Mat dist_coeffs_;
    };

}

#endif /* pnp_problem_hpp */
