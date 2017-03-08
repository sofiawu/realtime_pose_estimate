//
//  registration.cpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#include <stdio.h>
#include <iostream>

#include "mesh.hpp"
#include "robust_matcher.hpp"
#include "model_registration.hpp"
#include "utils.hpp"
#include "pnp_problem.hpp"
#include "model.hpp"

#include <opencv2/calib3d.hpp>

#include "estimate.h"

using namespace posest;

static double width = 1000, height = 750;
static double params_CANON[] = { 0.8638 * width,   // fx
    1.1527 * height,  // fy
    0.5012 * width - 0.5,      // cx
    0.4975 * height - 0.5};    // cy


static ModelRegistration registration;
static Mesh mesh;
static Model model;
static PnPProblem pnp_registration(params_CANON);

static bool end_registration = false;

// Some basic colors
static cv::Scalar red(0, 0, 255);
static cv::Scalar green(0,255,0);
static cv::Scalar blue(255,0,0);
static cv::Scalar yellow(0,255,255);

// Setup the points to register in the image
// In the order of the *.ply file and starting at 1
static const int n = 8;
static const int pts[] = {1, 2, 3, 4, 5, 6, 7, 8}; // 3 -> 4

// Mouse events for model registration
static void OnMouseModelRegistration( int event, int x, int y, int, void* ) {
    if  ( event == cv::EVENT_LBUTTONUP ) {
        int n_regist = registration.GetNumRegist();
        int n_vertex = pts[n_regist];
        
        cv::Point2f point_2d = cv::Point2f((float)x, (float)y);
        cv::Point3f point_3d = mesh.GetVertex(n_vertex - 1);
        
        bool is_registrable = registration.IsRegistrable();
        if (is_registrable) {
            registration.RegisterPoint(point_2d, point_3d);
            if( registration.GetNumRegist() == registration.GetNumMax() ) end_registration = true;
        }
    }
}

int CubeRegister(const std::string& img_path,
                 const std::string& ply_read_path,
                 const std::string& write_path) {
    // load a mesh given the *.ply file path
    mesh.Load(ply_read_path);
    
    // set parameters (train phase)
    int num_keypoints = 1000;
    
    //Instantiate robust matcher: detector, extractor, matcher
    RobustMatcher rmatcher;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(num_keypoints, 1.2f, 10);
    rmatcher.SetFeatureDetector(detector);
    
    
    /**  GROUND TRUTH OF THE FIRST IMAGE  **/
    
    // Create & Open Window
    cv::namedWindow("MODEL REGISTRATION", CV_WINDOW_KEEPRATIO);
    
    // Set up the mouse events
    cv::setMouseCallback("MODEL REGISTRATION", OnMouseModelRegistration, 0 );
    
    // Open the image to register
    cv::Mat img_in = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!img_in.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::Mat img_vis = img_in.clone();
    
    // Set the number of points to register
    int num_registrations = n;
    registration.SetNumMax(num_registrations);
    
    std::cout << "Click the box corners ..." << std::endl;
    std::cout << "Waiting ..." << std::endl;
    
    // Loop until all the points are registered
    while ( cv::waitKey(30) < 0 ) {
        // Refresh debug image
        img_vis = img_in.clone();
        
        // Current registered points
        std::vector<cv::Point2f> list_points2d = registration.GetPoints2d();
        std::vector<cv::Point3f> list_points3d = registration.GetPoints3d();
        
        // Draw current registered points
        DrawPoints(img_vis, list_points2d, list_points3d, red);
        
        // If the registration is not finished, draw which 3D point we have to register.
        // If the registration is finished, breaks the loop.
        if (!end_registration) {
            // Draw debug text
            int n_regist = registration.GetNumRegist();
            int n_vertex = pts[n_regist];
            cv::Point3f current_poin3d = mesh.GetVertex(n_vertex - 1);
            
            DrawQuestion(img_vis, current_poin3d, green);
            DrawCounter(img_vis, registration.GetNumRegist(), registration.GetNumMax(), red);
        } else {
            // Draw debug text
            DrawText(img_vis, "END REGISTRATION", green, cv::Point(25, 50));
            DrawCounter(img_vis, registration.GetNumRegist(), registration.GetNumMax(), green);
            break;
        }
        
        // Show the image
        imshow("MODEL REGISTRATION", img_vis);
    }
    
    /** COMPUTE CAMERA POSE **/
    
    std::cout << "COMPUTING POSE ..." << std::endl;
    
    // The list of registered points
    std::vector<cv::Point2f> list_points2d = registration.GetPoints2d();
    std::vector<cv::Point3f> list_points3d = registration.GetPoints3d();
    
    // Estimate pose given the registered points
    bool is_correspondence = pnp_registration.EstimatePose(list_points3d, list_points2d, CV_ITERATIVE);
    if ( is_correspondence ) {
        std::cout << "Correspondence found" << std::endl;
        
        // Compute all the 2D points of the mesh to verify the algorithm and draw it
        std::vector<cv::Point2f> list_points2d_mesh = pnp_registration.VerifyPoints(&mesh);
        Draw2DPoints(img_vis, list_points2d_mesh, green);
    } else {
        std::cout << "Correspondence not found" << std::endl << std::endl;
    }
    
    // Show the image
    cv::imshow("MODEL REGISTRATION", img_vis);
    
    // Show image until ESC pressed
    cv::waitKey(0);
    
    /** COMPUTE 3D of the image Keypoints **/
    
    // Containers for keypoints and descriptors of the model
    std::vector<cv::KeyPoint> keypoints_model;
    cv::Mat descriptors;
    
    // Compute keypoints and descriptors
    rmatcher.ComputeKeyPoints(img_in, keypoints_model);
    rmatcher.ComputeDescriptors(img_in, keypoints_model, descriptors);
    
    // Check if keypoints are on the surface of the registration image and add to the model
    for (unsigned int i = 0; i < keypoints_model.size(); ++i) {
        cv::Point2f point2d(keypoints_model[i].pt);
        cv::Point3f point3d;
        bool on_surface = pnp_registration.Backproject2DPoint(&mesh, point2d, point3d);
        if (on_surface) {
            model.AddCorrespondence(point2d, point3d);
            model.AddDescriptor(descriptors.row(i));
            model.AddKeypoint(keypoints_model[i]);
        } else
            model.AddOutlier(point2d);
    }
    
    // save the model into a *.yaml file
    //model.Save(write_path);
    model.SetModelType(1);
    model.SaveBinary(write_path);
    
    // Out image
    img_vis = img_in.clone();
    
    // The list of the points2d of the model
    std::vector<cv::Point2f> list_points_in = model.GetPoints2dIn();
    std::vector<cv::Point2f> list_points_out = model.GetPoints2dOut();
    
    // Draw some debug text
    std::string num = IntToString((int)list_points_in.size());
    std::string text = "There are " + num + " inliers";
    DrawText(img_vis, text, green, cv::Point(25, 50));
    
    // Draw some debug text
    num = IntToString((int)list_points_out.size());
    text = "There are " + num + " outliers";
    DrawText(img_vis, text, red, cv::Point(25, 75));
    
    // Draw the object mesh
    DrawObjectMesh(img_vis, &mesh, &pnp_registration, blue);
    
    // Draw found keypoints depending on if are or not on the surface
    Draw2DPoints(img_vis, list_points_in, green);
    Draw2DPoints(img_vis, list_points_out, red);
    
    // Show the image
    cv::imshow("MODEL REGISTRATION", img_vis);
    
    // Wait until ESC pressed
    cv::waitKey(0);
    
    // Close and Destroy Window
    cv::destroyWindow("MODEL REGISTRATION");
    
    std::cout << "GOODBYE" << std::endl;
    
    return 0;
}
