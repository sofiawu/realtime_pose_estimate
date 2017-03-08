//
//  main.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include <iostream>

#include "estimate.h"

static std::string img_path = "../../../data/box5.JPG";
static std::string ply_read_path = "../../../data/box.ply";
static std::string write_path = "../../../data/box.yml";

static std::string video_read_path = "../../../data/box.mp4";
static std::string video_write_path = "../../../data/box_output.mp4";

int main(int argc, const char * argv[]) {
    //register the cube
    //CubeRegister(img_path, ply_read_path, write_path);
    //detect the cube
    CubeDetection(write_path, ply_read_path, video_read_path, video_write_path);
    return 0;
}
