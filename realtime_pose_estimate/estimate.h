//
//  estimate.h
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef estimate_h
#define estimate_h
int CubeRegister(const std::string& img_path,
                        const std::string& ply_read_path,
                 const std::string& write_path);

int CubeDetection(const std::string& yml_read_path,
                  const std::string& ply_read_path,
                  const std::string& video_read_path,
                  const std::string& video_write_path);

#endif /* estimate_h */
