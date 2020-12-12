#include "ros/ros.h"
#include "include.h"
#include <sensor_msgs/PointCloud2.h>
#include "std_msgs/String.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Point32.h"
#include "detector3d/object.h"
#include "detector3d/objectsofonemat.h"
#include "pcl/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/filter.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"
#include "realsense2_camera/Extrinsics.h"
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "time.h"
#include "lidar_pointcloud.h"
#include "depth_pointcloud.h"
#include "detect.h"
//#include <image_transport/image_transport.h> // 用来发布和订阅图像信息
using namespace std;
Eigen::Matrix<float,3,3> MTR,MTR1,R_deptoimg,R_deptoimg1;//相机坐标旋转矩阵
Eigen::Vector3f V_T,V_T1,T_deptoimg,T_deptoimg1;//平移向量T
Eigen::Matrix<float,3,3> Inner_Transformation_Depth,InnerTransformation_Color,Inner_Transformation_Depth1,InnerTransformation_Color1;// 相机内参
cv::Mat Depthmate,Dec_mat,color_mat;
vector<string> classNamesVec;
const auto window_name= "RGB Image";
char* input="../input/VOC";
char* output="../output/";
std::string filename;
std::string weight="/home/mzc/code/CLionProjects/Yolov5_trt/engine/old/";
std::string classname_path="/home/mzc/code/CLionProjects/Yolov5_trt/engine/coco.names";
//stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
//////////////////////////////////////////
static float Data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
//for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
//    data[i] = 1.0;
static float prob[BATCH_SIZE * OUTPUT_SIZE];
IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
int inputIndex;
int outputIndex;
void* buffers[2];
cudaStream_t stream;
/////////////////////////////
//DNN opencv与CV-bridge冲突暂时不可用
/*Net net;//DNN net
using namespace cv::dnn;
const size_t inWidth      = 416;
const size_t inHeight     = 416;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 1/255.f;
const float meanVal       = 127.5;
std::vector<String> outNames;
String yolo_tiny_model ="/home/mzc/code/CLionProjects/DNN435/engine/enetb0-coco_final.weights";//yolov4.weights enetb0-coco_final.weights
String yolo_tiny_cfg =  "/home/mzc/code/CLionProjects/DNN435/engine/enet-coco.cfg";//yolov4.cfg enet-coco.cfg
 */
//////////////////////////////////////////////
class Camera{
public:
    bool Colorinfo= false;
    bool Depthinfo= false;
    bool Convertinfo= false;
    bool Colorinfo1= false;
    bool Depthinfo1= false;
    bool Convertinfo1= false;
    int camid,camid_d;
    lidar_pointcloud lidar;
    Objects Objectlidartoimg,ObjectDepthtoimg,ObjectDepthtoimg1;
    std_msgs::Header this_head;
    std_msgs::Header cloudHeader;
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn;
    ros::NodeHandle n;
    ros::Subscriber subrgbcaminfo,subrgbcaminfo1;
    ros::Subscriber subdepthcaminfo,subdepthcaminfo1;
    ros::Subscriber subrgbimg,subrgbimg1;
    ros::Subscriber subdepthimg,subdepthimg1 ;
    ros::Subscriber subdepthtoclolor ;
    ros::Subscriber sublidarpcl;
    ros::Publisher Object_pub,Objectimg_pub1;
    ros::Publisher Objectimg_pub,clusterimage_pub,clusterimgdp_pub;
    ros::Publisher pointcloud_clustered,pointcloud_camera,pointcloud_detect,pointcloud_detect1;
    Camera(){
        if (LidarEnable)
            sublidarpcl=n.subscribe<sensor_msgs::PointCloud2>("/rslidar_points",1, &Camera::pointcloudcallback, this);//rslidar_points
        subrgbcaminfo= n.subscribe("/D455/color/camera_info", 10, &Camera::inforgbcallback, this);
        subrgbcaminfo1= n.subscribe("/Astra/rgb/camera_info", 10, &Camera::inforgbcallback1, this);

        if (DepthEnable){
            R_deptoimg<<1,0,0,0,1,0,0,0,1;//初始化深度-彩色相机 相机转换矩阵 可能未发布
            T_deptoimg<<0,0,0;
            R_deptoimg1<<1,0,0,0,1,0,0,0,1;
            T_deptoimg1<<0,0,0;
            subdepthtoclolor = n.subscribe( "/D455/extrinsics/depth_to_color",1, &Camera::depth_to_colorcallback,this);
            subdepthcaminfo = n.subscribe("/D455/depth/camera_info", 1, &Camera::infodepthcallback,this);
            subdepthimg = n.subscribe("/D455/depth/image_rect_raw", 10, &Camera::imgdepthcallback,this);
            subdepthcaminfo1 = n.subscribe("/Astra/depth/camera_info", 1, &Camera::infodepthcallback1,this);
            subdepthimg1 = n.subscribe("/Astra/depth/image_rect_raw", 10, &Camera::imgdepthcallback,this);
        }
        subrgbimg= n.subscribe("/D455/color/image_raw", 10, &Camera::imgrgbcallback,this);
        subrgbimg1= n.subscribe("/Astra/rgb/image_raw", 10, &Camera::imgrgbcallback,this);

        if (NORT)
        {
            MTR<<0,-1,0,0,0,-1,1,0,0;//深度相机和RGB相机转换外参x->-y y->-z z->x前方相机 realsense
            V_T<<0.0965,-0.01,-0.058;//baselink在相机坐标系下坐标 用于雷达像相机投影
            MTR1<<0,1,0,0,0,-1,-1,0,0;//深度相机和RGB相机转换外参x->y y->-z z->-x  后方相机 Astra
            V_T1<<-0.1265,-0.01,-0.058;//baselink在相机坐标系下坐标 用于雷达像相机投影
        }
        if (LidarEnable){
            pointcloud_camera=n.advertise<sensor_msgs::PointCloud2>("/pointcloud_camera",1);
        }
        pointcloud_clustered=n.advertise<sensor_msgs::PointCloud2>("/pointcloud_clustered",1);
        pointcloud_detect=n.advertise<sensor_msgs::PointCloud2>("/pointcloud_detect",1);
        pointcloud_detect1=n.advertise<sensor_msgs::PointCloud2>("/pointcloud_detect1",1);
//        Object_pub =n.advertise<detector3d::objectsofonemat>("Objects",10);
        Objectimg_pub=n.advertise<sensor_msgs::Image>("Objectsimg",1);
        Objectimg_pub1=n.advertise<sensor_msgs::Image>("Objectsimg1",1);

    }

    //订阅相机内参数
    void inforgbcallback(const sensor_msgs::CameraInfo &caminfo){//相机1 realsense D455
        if (Colorinfo)
            return;
        else{
            ///获取彩色相机内参
            std::cout<<"\ncolor intrinsics D455: "<<endl;
            InnerTransformation_Color<<caminfo.K.at(0),caminfo.K.at(1),caminfo.K.at(2),caminfo.K.at(3),caminfo.K.at(4),caminfo.K.at(5),caminfo.K.at(6),caminfo.K.at(7),caminfo.K.at(8);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout<<InnerTransformation_Color(i,j)<<"\t";
                }
                std::cout<<std::endl;
            }
            lidar.initcamera(V_T,MTR,InnerTransformation_Color,caminfo.height,caminfo.width,0);
            Colorinfo= true;
        }
    }
    void inforgbcallback1(const sensor_msgs::CameraInfo &caminfo){//相机2 Astra
        if (Colorinfo1)
            return;
        else{
            ///获取彩色相机内参
            std::cout<<"\ncolor intrinsics Astra: "<<endl;
            InnerTransformation_Color1<<caminfo.K.at(0),caminfo.K.at(1),caminfo.K.at(2),caminfo.K.at(3),caminfo.K.at(4),caminfo.K.at(5),caminfo.K.at(6),caminfo.K.at(7),caminfo.K.at(8);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout<<InnerTransformation_Color1(i,j)<<"\t";
                }
                std::cout<<std::endl;
            }
            lidar.initcamera(V_T1,MTR1,InnerTransformation_Color1,caminfo.height,caminfo.width,1);
            Colorinfo1= true;
        }
    }
    void infodepthcallback(const sensor_msgs::CameraInfo &caminfo){
        if (Depthinfo)
            return;
        else{
            //获取深度相机内参
            std::cout<<"\ndepth intrinsics: D455 "<<endl;
            Inner_Transformation_Depth<<caminfo.K.at(0),caminfo.K.at(1),caminfo.K.at(2),caminfo.K.at(3),caminfo.K.at(4),caminfo.K.at(5),caminfo.K.at(6),caminfo.K.at(7),caminfo.K.at(8);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout<<Inner_Transformation_Depth(i,j)<<"\t";
                }
                std::cout<<std::endl;
            }
            Depthinfo= true;
        }
    }
    void infodepthcallback1(const sensor_msgs::CameraInfo &caminfo){
        if (Depthinfo1)
            return;
        else{
            //获取深度相机内参
            std::cout<<"\ndepth intrinsics: Astra"<<endl;
            Inner_Transformation_Depth1<<caminfo.K.at(0),caminfo.K.at(1),caminfo.K.at(2),caminfo.K.at(3),caminfo.K.at(4),caminfo.K.at(5),caminfo.K.at(6),caminfo.K.at(7),caminfo.K.at(8);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    std::cout<<Inner_Transformation_Depth1(i,j)<<"\t";
                }
                std::cout<<std::endl;
            }
            Depthinfo1= true;
        }
    }
    void depth_to_colorcallback(const realsense2_camera::Extrinsics &extrin) {
        if (Convertinfo)
            return;
        else{
            //获取深度相机相对于彩色相机的外参，即变换矩阵: P_color = R * P_depth + T
            std::cout<<"\nextrinsics of depth camera to color camera: \nrotaion: "<<std::endl;
            R_deptoimg<<extrin.rotation[0],extrin.rotation[1],extrin.rotation[2],extrin.rotation[3],extrin.rotation[4],extrin.rotation[5],extrin.rotation[6],extrin.rotation[7],extrin.rotation[8];
            for(int i = 0; i < 3; ++i){
                for(int j = 0; j < 3; ++j){
                    std::cout<<R_deptoimg(i,j)<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
            std::cout<<"translation: ";
            T_deptoimg<<extrin.translation[0],extrin.translation[1],extrin.translation[2];
            for(int i=0;i<3;i++)
                std::cout<<T_deptoimg(i)<<"\t";
            std::cout<<std::endl;
            Convertinfo= true;
        }
    }




    //点云处理回调函数
    void pointcloudcallback(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        if (Colorinfo==false||Colorinfo1==false)
            return;
        lidar.vec_clu.clear();//
        sensor_msgs::PointCloud2 clustered,camera;
        cloudHeader.stamp=camera.header.stamp=clustered.header.stamp=laserCloudMsg->header.stamp;
        laserCloudIn.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
         lidar.init(laserCloudIn);
        lidar.pointcloud_cluster();
        pcl::toROSMsg(*(lidar.cloud_cluster),clustered);
        pcl::toROSMsg(*(lidar.cloudIncamera),camera);
        clustered.header.frame_id="base_link";
        camera.header.frame_id="base_link";
        pointcloud_clustered.publish(clustered);//发布聚类后的点云 intensity的值表示类别
        pointcloud_camera.publish(camera);
    }
    //图像处理回调函数
    void imgrgbcallback(const sensor_msgs::ImageConstPtr& msg){
        if (Colorinfo== false||Colorinfo1== false)
            return;
        auto Timeofimg=msg->header.stamp.toSec()-cloudHeader.stamp.toSec();
//        if (LidarEnable){
//            if (std::abs(Timeofimg)>=0.1)//时间戳同步
//            {
//            cout<<"TimeOF="<<Timeofimg<<endl;
//                return;
//            }
//        }
        if (msg->header.frame_id=="Astra_rgb_optical_frame")
            camid=1;
        if (msg->header.frame_id=="D455_color_optical_frame")
            camid=0;
        if (LidarEnable){
            Objectlidartoimg.VecIOU.clear();//清空原始目标的容器
            Objectlidartoimg.objects.clear();
            vector<Cluster> clus;
            if (!lidar.vec_clu.empty())
            {
                for (auto clu:lidar.vec_clu) {
                    if (clu.index == camid)
                        clus.push_back(clu);
                }
            }
            Objectlidartoimg.clusters = clus;
        }
        if (DepthEnable){
            ObjectDepthtoimg.VecIOU.clear();
            ObjectDepthtoimg1.VecIOU.clear();
            ObjectDepthtoimg.objects.clear();
            ObjectDepthtoimg1.objects.clear();
        }
        vector<cv::Rect> DetectBox;
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        color_mat=cv_ptr->image;
        Dec_mat = Dection(color_mat);
        sensor_msgs::ImagePtr img=cv_bridge::CvImage(std_msgs::Header(),"bgr8",Dec_mat).toImageMsg();
        img->header.stamp=ros::Time::now();
        img->header.frame_id="base_link";
        if (camid==0)
            Objectimg_pub.publish(img);//发布检测后的二维目标检测图像
        else if (camid==1)
            Objectimg_pub1.publish(img);
            // 点云信号发布
        pcl::PointCloud<pcl::PointXYZI> cloud;
        if (DepthEnable) {
            Objects ObjectDTI;
            if (camid==0)
                ObjectDTI=ObjectDepthtoimg;
            if (camid==1)
                ObjectDTI=ObjectDepthtoimg1;
            if(ObjectDTI.objects.size()>0&&ObjectDTI.clusters.size()>0)
            {
                ObjectDTI.BoxesMatch();
                ObjectDTI.initobjects();
                for (auto object:ObjectDTI.objects) {
                    pcl::PointXYZI point;
                    for (auto i:object.Pointcloud->points) {
                        point.x = i.x;
                        point.y = i.y;
                        point.z = i.z;
                        point.intensity = 255 - object.object.classID * 10;//Intensity 用于可视化区分类别
//                  cout<<"point="<<i.x<<i.y<<i.z<<endl;
                        cloud.push_back(point);
                    }
                }
            }
        }
        if (LidarEnable&&Objectlidartoimg.objects.size()>0&&Objectlidartoimg.clusters.size()>0){
            Objectlidartoimg.BoxesMatch();//匹配摄像头和雷达的点云
            Objectlidartoimg.initobjects();//获取语义点云
            for (auto object:Objectlidartoimg.objects){
               pcl::PointXYZI point;
                for(auto i:object.Pointcloud->points){
                  point.x=i.x;
                  point.y=i.y;
                  point.z=i.z;
                  point.intensity = 255-object.object.classID*10;//Intensity 用于可视化区分类别
//                  cout<<"point="<<i.x<<i.y<<i.z<<endl;
                  cloud.push_back(point);
                }
            }
        }
        sensor_msgs::PointCloud2 PointcloudDetect;
        pcl::toROSMsg(cloud,PointcloudDetect);
        PointcloudDetect.header.stamp=ros::Time::now();
        PointcloudDetect.header.frame_id="base_link";
        if(camid==0)
            pointcloud_detect.publish(PointcloudDetect);
        else if(camid==1)
            pointcloud_detect1.publish(PointcloudDetect);
    }
    //深度流处理回调函数
    void imgdepthcallback(const sensor_msgs::ImageConstPtr& msg){
        if(Depthinfo==0||Depthinfo1==0)
            return;
        cloudHeader.stamp = msg->header.stamp;
        if (msg->header.frame_id=="Astra_rgb_optical_frame")
            camid_d=1;
        if (msg->header.frame_id=="D455_depth_optical_frame")
            camid_d=0;
        if (camid_d==0)
            ObjectDepthtoimg.clusters.clear();
        if (camid_d==1)
            ObjectDepthtoimg1.clusters.clear();
        sensor_msgs::PointCloud2 clustered;
        depth_pointcloud depth;
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        Depthmate=cv_ptr->image;
//        clustered.header=this_head=msg->header;
        depth.init(Depthmate,camid_d);
        depth.pointcloud_cluster();
        pcl::toROSMsg(*(depth.cloud_cluster),clustered);
        clustered.header.stamp=ros::Time::now();
        clustered.header.frame_id="base_link";
        pointcloud_clustered.publish(clustered);//发布聚类后的点云 intensity的值表示类别
        if (camid_d==0)
            ObjectDepthtoimg.clusters=depth.vec_clu;
        if(camid_d==1)
            ObjectDepthtoimg1.clusters=depth.vec_clu;
    }
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
    cv::Mat Dection(cv::Mat img){
        int fcount = 0;
        fcount++;
        for (int b = 0; b < fcount; b++) {
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar *uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    Data[b * 3 * INPUT_H * INPUT_W + i] = (float) uc_pixel[2] / 255.0;
                    Data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float) uc_pixel[1] / 255.0;
                    Data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float) uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, Data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            ObjectImgtoDepth ObjectItoD;
            std::cout <<"num:"<<res.size()<<" ";
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, classNamesVec[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                std::cout<<classNamesVec[(int)res[j].class_id]<<";";
                if (DepthEnable){
                    ObjectItoD.object.classID=(int)res[j].class_id;
                    ObjectItoD.object.classname=classNamesVec[(int)res[j].class_id];
                    if (r.x<0)
                        r.x=0;
                    if (r.y<0)
                        r.y=0;
                    ObjectItoD.Rimg=r;
                    if (camid==0)
                        ObjectDepthtoimg.objects.push_back(ObjectItoD);
                    if (camid==1)
                        ObjectDepthtoimg1.objects.push_back(ObjectItoD);
                }
                if (LidarEnable){
                    ObjectItoD.object.classID=(int)res[j].class_id;
                    ObjectItoD.object.classname=classNamesVec[(int)res[j].class_id];
                    if (r.x<0)
                        r.x=0;
                    if (r.y<0)
                        r.y=0;
                    ObjectItoD.Rimg=r;
                    Objectlidartoimg.objects.push_back(ObjectItoD);
                }
            }
            std::cout<<std::endl;
        }
        return img;
    }
};



int main(int argc, char** argv)
{
//    image_detection_Cfg();//DNN二维目标检测初始化
//////////////TensorRT//////////////////
    cudaSetDevice(DEVICE);
    std::vector<std::string> file_names;
    clock_t last_time;
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = STR2(NET);
    engine_name = weight+"yolov5" + engine_name + ".engine";
    std::ifstream file(engine_name);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::ifstream classNamesFile(classname_path);
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    // prepare input data ---------------------------
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr)  ;
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
    //////////////TensorRT载入模型////////////////////
    ros::init(argc,argv,"object_detect");
    ///////////////
    Camera CA;
    ////////////////////////
    ros::spin();
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;

}
//////////////////////////Opencv 版本冲突 暂时无法使用DNN
/*
void image_detection_Cfg() {
    net = readNetFromDarknet(yolo_tiny_cfg, yolo_tiny_model);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);// DNN_BACKEND_INFERENCE_ENGINE DNN_BACKEND_OPENCV 未安装IE库 if you have IntelCore CPU you can chose this Para to accelerate youe model--Openvino;
    net.setPreferableTarget(DNN_TARGET_CPU);
    outNames = net.getUnconnectedOutLayersNames();
    for (int i = 0; i < outNames.size(); i++) {
        printf("output layer name : %s\n", outNames[i].c_str());
    }
    ifstream classNamesFile(classname_path);
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
}
Mat Dectection(Mat &color_mat) {  //getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0
    // 加载图像 D435加载
//        cout<<color_mat.type()<<endl;
    Mat inputBlob = blobFromImage(color_mat, inScaleFactor, Size(inWidth, inHeight), Scalar(), true, false);
    net.setInput(inputBlob);
    // 检测
    std::vector<Mat> outs;
    net.forward(outs, outNames);
    vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;
    double FPS=1000/time;
    ostringstream ss;
    ss << "FPS: " << FPS ;
    putText(color_mat, ss.str(), Point(0, 10), 0, 0.5, Scalar(255, 0, 0));
    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;
    for (size_t i = 0; i < outs.size(); ++i) {
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5) {
                int centerX = (int) (data[0] * color_mat.cols);
                int centerY = (int) (data[1] * color_mat.rows);
                int width = (int) (data[2] * color_mat.cols);
                int height = (int) (data[3] * color_mat.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        auto ClassID=classIds[idx];
        String className = classNamesVec[classIds[idx]];
        Objection NewObjection(box,ClassID);
        ObjectionOfOneMat.push_back(NewObjection);
        putText(color_mat, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
        rectangle(color_mat, box, Scalar(0, 0, 255), 2, 8, 0);
    }
    return color_mat;
}
*/
/////////////////////////////////////////////////