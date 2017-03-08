//
//  robust_matcher.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef robust_matcher_hpp
#define robust_matcher_hpp

#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace posest {
    
    class RobustMatcher {
    public:
        
        RobustMatcher() : ratio_(0.8f) {
            // ORB is the default feature
            detector_ = cv::ORB::create();
            extractor_ = cv::ORB::create();
            
            // BruteFroce matcher with Norm Hamming is the default matcher
            matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_HAMMING, false);
            
        }
        virtual ~RobustMatcher();
        
        // Set the feature detector
        inline void SetFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {  detector_ = detect; }
        // Set the descriptor extractor
        inline void SetDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) { extractor_ = desc; }
        // Set the matcher
        inline void SetDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher>& match) {  matcher_ = match; }
        
        // Set ratio parameter for the ratio test
        inline void SetRatio( float rat) { ratio_ = rat; }
        
        // Compute the keypoints of an image
        void ComputeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
        // Compute the descriptors of an image given its keypoints
        void ComputeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        
        
        // Clear matches for which NN ratio is > than threshold
        // return the number of removed points
        // (corresponding entries being cleared,
        // i.e. size will be 0)
        int RatioTest(std::vector<std::vector<cv::DMatch> > &matches);
        
        // Insert symmetrical matches in symMatches vector
        void SymmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                          const std::vector<std::vector<cv::DMatch> >& matches2,
                          std::vector<cv::DMatch>& symMatches );
        
        // Match feature points using ratio and symmetry test
        void RobustMatch( const cv::Mat& frame,
                         std::vector<cv::DMatch>& good_matches,
                         std::vector<cv::KeyPoint>& keypoints_frame,
                         const cv::Mat& descriptors_model );
        
        // Match feature points using ratio test
        void FastRobustMatch( const cv::Mat& frame,
                             std::vector<cv::DMatch>& good_matches,
                             std::vector<cv::KeyPoint>& keypoints_frame,
                             const cv::Mat& descriptors_model );
        
    private:
        // pointer to the feature point detector object
        cv::Ptr<cv::FeatureDetector> detector_;
        // pointer to the feature descriptor extractor object
        cv::Ptr<cv::DescriptorExtractor> extractor_;
        // pointer to the matcher object
        cv::Ptr<cv::DescriptorMatcher> matcher_;
        // max ratio between 1st and 2nd NN
        float ratio_;
    };

}

#endif /* robust_matcher_hpp */
