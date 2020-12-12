//
// Created by mzc on 2020/11/28.
//

#include "lidar_pointcloud.h"
#include "cv_bridge/cv_bridge.h"
lidar_pointcloud::lidar_pointcloud() {
}
void lidar_pointcloud::initcamera(Eigen::Vector3f T,Eigen::Matrix<float,3,3> R,Eigen::Matrix<float,3,3> inner,int H,int W,int id){
    Camera_info camtemp;
    camtemp.TLtoC=T;
    camtemp.RLtoC=R;
    camtemp.Camerainfo=inner;
    camtemp.imagewidth=W;
    camtemp.imageheigth=H;
    camtemp.index=id;
    cam_vec.push_back(camtemp);
}
void lidar_pointcloud::init(pcl::PointCloud<pcl::PointXYZI>::Ptr pointclouds){
    cloud_cluster.reset(new pcl::PointCloud<pcl::PointXYZI>);
    cloudIncamera.reset(new pcl::PointCloud<pcl::PointXYZI>);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    tree.reset(new pcl::search::KdTree<pcl::PointXYZI>);
    for (auto point:pointclouds->points) {
        if (point.z>-0.3)
            cloud->push_back(point);
    }
    cloudsize=cloud->size();
    cluster_indices.clear();
}
template <typename PointCloudPtrType>
void lidar_pointcloud:: show_point_cloud(PointCloudPtrType cloud, std::string display_name) {
    pcl::visualization::CloudViewer viewer(display_name);
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}
void lidar_pointcloud::pointcloud_cluster(){
//    show_point_cloud(cloud, "original pointcloud");
//    lidar_clus=cv::Mat {imageheigth,imagewidth,CV_8UC1};
//    lidar_depth=cv::Mat {imageheigth,imagewidth,CV_16UC1};
    clock_t start_ms = clock();
//    lidar_depth=cv::Mat::zeros(cam_vec.at(0).imageheigth,cam_vec.at(0).imagewidth,CV_16UC1);
//    lidar_clus=cv::Mat::zeros(cam_vec.at(0).imageheigth,cam_vec.at(0).imagewidth,CV_8UC1);
    int j = 0;
    Eigen::Vector3f point3d;
    Cluster Clu_tmp;
    for(auto point:cloud->points){
        point3d.x() = point.x;
        point3d.y() = point.y;
        point3d.z() = point.z;
        for (auto cam:cam_vec) {
            auto B=cam.Camerainfo*cam.RLtoC*(point3d+cam.TLtoC);
            Eigen::Vector3f imgpix=B/B.z();
//            Eigen::Vector3f imgpix=Camerainfo*RLtoC*(point3d-TLtoC)/tmp.z;
            if (imgpix.x()>=0&&imgpix.y()>=0&&B.z()>0&&imgpix.x()<cam.imagewidth-1&&imgpix.y()<cam.imageheigth-1) {
                cloudIncamera->push_back(point);
            }
        }
    }
    // KdTree, for more information, please ref [How to use a KdTree to search](https://pcl.readthedocs.io/projects/tutorials/en/latest/kdtree_search.html#kdtree-search)
    tree->setInputCloud(cloudIncamera);
    // test 1. uncomment the following two lines to test the simple dbscan
    // DBSCANSimpleCluster<pcl::PointXYZI> ec;
    // ec.setCorePointMinPts(20);

    // test 2. uncomment the following two lines to test the precomputed dbscan
    // DBSCANPrecompCluster<pcl::PointXYZI>  ec;
    // ec.setCorePointMinPts(20);

    // test 3. uncomment the following two lines to test the dbscan with Kdtree for accelerating
    ec.setCorePointMinPts(20);

    // test 4. uncomment the following line to test the EuclideanClusterExtraction

    ec.setClusterTolerance(0.2);
    ec.setMinClusterSize(20);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloudIncamera);
    ec.extract(cluster_indices);
//    std::cout<<cluster_indices.end()-cluster_indices.begin()<<std::endl;
    //visualization, use indensity to show different color for each cluster.
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++, j++) {
        for(auto cam:cam_vec){
            int minx=cam.imagewidth,miny=cam.imageheigth,maxx=0,maxy=0;
            Clu_tmp.pointcould.reset(new pcl::PointCloud<pcl::PointXYZI>);
            for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
                pcl::PointXYZI tmp;
                pcl::PointXYZI tmpn;
                //雷达坐标系到相机坐标系坐标轴变化
                point3d.x()=tmpn.x=tmp.x = cloudIncamera->points[*pit].x;

                point3d.y()=tmpn.y=tmp.y = cloudIncamera->points[*pit].y;

                point3d.z()=tmpn.z=tmp.z = cloudIncamera->points[*pit].z;
                tmp.intensity = j;
                cloud_cluster->points.push_back(tmp);
                auto B=cam.Camerainfo*cam.RLtoC*(point3d+cam.TLtoC);
                Eigen::Vector3f imgpix=B/B.z();
//            Eigen::Vector3f imgpix=Camerainfo*RLtoC*(point3d-TLtoC)/tmp.z;
//            if (imgpix.x()>=0&&imgpix.y()>=0&&B.z()>0&&imgpix.x()<imagewidth-1&&imgpix.y()<imageheigth-1) {
//                std::cout<<"imgpix:"<<imgpix.x()<<" "<<imgpix.y()<<" "<<imgpix.z()<<std::endl;
                if (minx>imgpix.x())
                    minx=imgpix.x();
                if (miny>imgpix.y())
                    miny=imgpix.y();
                if (maxx<imgpix.x())
                    maxx=imgpix.x();
                if (maxy<imgpix.y())
                    maxy=imgpix.y();
                Clu_tmp.pointcould->push_back(tmpn);
//                lidar_clus.at<uint8_t>(static_cast<int>(imgpix.y()), static_cast<int>(imgpix.x())) = j;//-j*255/(cluster_indices.end()-cluster_indices.begin());
//                lidar_depth.at<uint16_t>(static_cast<int>(imgpix.y()), static_cast<int>(imgpix.x())) = static_cast<uint16_t>(imgpix.z());
//            }
            }
            if (!Clu_tmp.pointcould->empty()){
                Clu_tmp.index=cam.index;//标记相机编号
                Clu_tmp.cloudsize=Clu_tmp.pointcould->size();
                Clu_tmp.R=cv::Rect(minx,miny,(maxx-minx),(maxy-miny));
                vec_clu.push_back(Clu_tmp);
//            std::cout<<"RECT:"<<index<<":<"<<minx<<","<<miny<<"><"<<maxx<<","<<maxy<<"> size:"<<Clu_tmp.cloudsize<<std::endl;
            }
        }
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height = 1;
//    show_point_cloud(cloud_cluster, "colored clusters of point cloud");//可视化
    clock_t end_ms = clock();
    std::cout << "cluster time cost:" << double(end_ms - start_ms) / CLOCKS_PER_SEC << " s" << std::endl;
}
