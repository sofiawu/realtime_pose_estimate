//
//  Mesh.hpp
//  realtime_pose_estimate
//
//  Created by sofiawu on 2017/3/8.
//  Copyright © 2017年 sofiawu. All rights reserved.
//

#ifndef Mesh_hpp
#define Mesh_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>

namespace posest {
    // --------------------------------------------------- //
    //                 TRIANGLE CLASS                      //
    // --------------------------------------------------- //
    class Triangle {
    public:
        
        explicit Triangle(int id, cv::Point3f V0, cv::Point3f V1, cv::Point3f V2);
        virtual ~Triangle();
        
        inline cv::Point3f GetV0() const { return v0_; }
        inline cv::Point3f GetV1() const { return v1_; }
        inline cv::Point3f GetV2() const { return v2_; }
        
    private:
        /** The identifier number of the triangle */
        int id_;
        /** The three vertices that defines the triangle */
        cv::Point3f v0_, v1_, v2_;
    };
    
    // --------------------------------------------------- //
    //                     RAY CLASS                       //
    // --------------------------------------------------- //
    class Ray {
    public:
        
        explicit Ray(cv::Point3f P0, cv::Point3f P1);
        virtual ~Ray();
        
        inline cv::Point3f GetP0() const { return p0_; }
        inline cv::Point3f GetP1() const { return p1_; }
        
    private:
        /** The two points that defines the ray */
        cv::Point3f p0_, p1_;
    };
    
    // --------------------------------------------------- //
    //                OBJECT MESH CLASS                    //
    // --------------------------------------------------- //
    class Mesh {
    public:
        
        Mesh();
        virtual ~Mesh();
        
        inline std::vector<std::vector<int> > GetTrianglesList() const { return list_triangles_; }
        inline cv::Point3f GetVertex(int pos) const { return list_vertex_[pos]; }
        inline int GetNumVertices() const { return num_vertexs_; }
        
        void Load(const std::string path_file);
        
    private:
        /** The identification number of the mesh */
        int id_;
        /** The current number of vertices in the mesh */
        int num_vertexs_;
        /** The current number of triangles in the mesh */
        int num_triangles_;
        /* The list of triangles of the mesh */
        std::vector<cv::Point3f> list_vertex_;
        /* The list of triangles of the mesh */
        std::vector<std::vector<int> > list_triangles_;
    };

}

#endif /* Mesh_hpp */
