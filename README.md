# Detect3D-Lidar-DepthCamera
基于聚类方法的2D到3D目标检测方法 包含密度聚类和半监督PCKMeans半监督聚类 Yolov5目标检测方法--TensorRT加速
支持多线激光雷达+RGB相机、深度相机两种方式
ROS下编译运行
模型选择位于src/detect.cpp中 可更换自己的目标检测模型 但需要在TensorRT下支持
```cpp
std::string weight="/home/ax/code/Yolov5_trt/engine/";
std::string classname_path="/home/ax/code/Yolov5_trt/engine/coco.names";
``` 
模式选择位位于include/include.h中 可选雷达 或深度相机模式
```cpp
#define DepthEnable 1
#define LidarEnable 0
``` 
## 演示视频
在Lego-LOAM下机器人运行效果
[![Watch the video](https://github.com/Mazhichaoruya/Perception-of-Autonomous-mobile-robot/blob/master/image/image.png)](https://www.bilibili.com/video/BV1QK4y1V7fy)
## 其他
待更新～