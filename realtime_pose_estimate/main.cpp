//
//  main.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include <iostream>

#include "estimate.h"

static std::string cube_img_path = "../../../data/box5.JPG";
static std::string cube_ply_read_path = "../../../data/box.ply";
static std::string cube_write_path = "../../../data/box.yml";
static std::string cube_write_binary_path = "../../../data/box.bin";

static std::string cube_video_read_path = "../../../data/box.mp4";
static std::string cube_video_write_path = "../../../data/box_output.mp4";

static std::string cylinder_img_path = "../../../data/cylinder_side.png";
static std::string cylinder_ply_read_path = "../../../data/cylinder.ply";
static std::string cylinder_write_path = "../../../data/cylinder.yml";
static std::string cylinder_write_binary_path = "../../../data/cylinder.bin";

static std::string cylinder_video_read_path = "../../../data/cylinder.MOV";
static std::string cylinder_video_write_path = "../../../data/cylinder_output1.MOV";


int main(int argc, const char * argv[]) {
    //register the cube
    //CubeRegister(img_path, ply_read_path, write_binary_path);
    //detect the cube
    //CubeDetection(write_binary_path, ply_read_path, video_read_path, video_write_path);
    
    CylinderRegister(cylinder_img_path, cylinder_write_binary_path);
    CylinderDetection(cylinder_write_binary_path, cylinder_video_read_path, cylinder_video_write_path);
    
    return 0;
}
