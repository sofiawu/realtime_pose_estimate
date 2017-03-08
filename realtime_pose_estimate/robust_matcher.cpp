//
//  robust_matcher.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include "robust_matcher.hpp"

namespace posest {
    
    RobustMatcher::~RobustMatcher()
    {
    }
    
    void RobustMatcher::ComputeKeyPoints( const cv::Mat& image,
                                         std::vector<cv::KeyPoint>& keypoints) {
        detector_->detect(image, keypoints);
    }
    
    void RobustMatcher::ComputeDescriptors( const cv::Mat& image,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors) {
        extractor_->compute(image, keypoints, descriptors);
    }
    
    int RobustMatcher::RatioTest(std::vector<std::vector<cv::DMatch> > &matches) {
        int removed = 0;
        // for all matches
        for ( auto iter = matches.begin(); iter != matches.end(); ++iter) {
            // if 2 NN has been identified
            if (iter->size() > 1) {
                // check distance ratio
                if ((*iter)[0].distance / (*iter)[1].distance > ratio_) {
                    iter->clear(); // remove match
                    removed++;
                }
            } else { // does not have 2 neighbours
                iter->clear(); // remove match
                removed++;
            }
        }
        return removed;
    }
    
    void RobustMatcher::SymmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                                     const std::vector<std::vector<cv::DMatch> >& matches2,
                                     std::vector<cv::DMatch>& sym_matches ) {
        // for all matches image 1 -> image 2
        for (auto iter1 = matches1.begin(); iter1 != matches1.end(); ++iter1) {
            
            // ignore deleted matches
            if (iter1->empty() || iter1->size() < 2) continue;
            
            // for all matches image 2 -> image 1
            for (auto iter2 = matches2.begin(); iter2 != matches2.end(); ++iter2) {
                // ignore deleted matches
                if (iter2->empty() || iter2->size() < 2) continue;
                
                // Match symmetry test
                if ((*iter1)[0].queryIdx == (*iter2)[0].trainIdx &&
                    (*iter2)[0].queryIdx == (*iter1)[0].trainIdx) {
                    // add symmetrical match
                    sym_matches.push_back( cv::DMatch((*iter1)[0].queryIdx,
                                                    (*iter1)[0].trainIdx,
                                                    (*iter1)[0].distance));
                    break; // next match in image 1 -> image 2
                }
            }
        }
        
    }
    
    void RobustMatcher::RobustMatch( const cv::Mat& frame,
                                    std::vector<cv::DMatch>& good_matches,
                                    std::vector<cv::KeyPoint>& keypoints_frame,
                                    const cv::Mat& descriptors_model ) {
        
        // 1a. Detection of the ORB features
        this->ComputeKeyPoints(frame, keypoints_frame);
        
        // 1b. Extraction of the ORB descriptors
        cv::Mat descriptors_frame;
        this->ComputeDescriptors(frame, keypoints_frame, descriptors_frame);
        
        // 2. Match the two image descriptors
        std::vector<std::vector<cv::DMatch> > matches12, matches21;
        
        // 2a. From image 1 to image 2
        matcher_->knnMatch(descriptors_frame, descriptors_model, matches12, 2); // return 2 nearest neighbours
        
        // 2b. From image 2 to image 1
        matcher_->knnMatch(descriptors_model, descriptors_frame, matches21, 2); // return 2 nearest neighbours
        
        // 3. Remove matches for which NN ratio is > than threshold
        // clean image 1 -> image 2 matches
        RatioTest(matches12);
        // clean image 2 -> image 1 matches
        RatioTest(matches21);
        
        // 4. Remove non-symmetrical matches
        SymmetryTest(matches12, matches21, good_matches);
    }
    
    void RobustMatcher::FastRobustMatch( const cv::Mat& frame,
                                        std::vector<cv::DMatch>& good_matches,
                                        std::vector<cv::KeyPoint>& keypoints_frame,
                                        const cv::Mat& descriptors_model ) {
        good_matches.clear();
        
        // 1a. Detection of the ORB features
        this->ComputeKeyPoints(frame, keypoints_frame);
        
        // 1b. Extraction of the ORB descriptors
        cv::Mat descriptors_frame;
        this->ComputeDescriptors(frame, keypoints_frame, descriptors_frame);
        
        // 2. Match the two image descriptors
        std::vector<std::vector<cv::DMatch> > matches;
        matcher_->knnMatch(descriptors_frame, descriptors_model, matches, 2);
        
        // 3. Remove matches for which NN ratio is > than threshold
        RatioTest(matches);
        
        // 4. Fill good matches container
        for ( auto iter = matches.begin(); iter!= matches.end(); ++iter)
        {
            if (!iter->empty()) good_matches.push_back((*iter)[0]);
        }
    }

}
