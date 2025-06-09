#pragma once
#include <opencv2/opencv.hpp>
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};