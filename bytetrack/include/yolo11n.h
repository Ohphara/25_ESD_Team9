#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "object_type.h"
// yolo11n.cpp에서 구현
int detect_yolo11(const cv::Mat& bgr, std::vector<Object>& objects);