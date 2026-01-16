#pragma once

#include <opencv2/core/core.hpp>

#include "mat.h"

namespace yolo26 {

struct LetterBoxInfo {
    float gain = 1.f;  // resize ratio r
    int pad_x = 0;     // left padding in pixels
    int pad_y = 0;     // top padding in pixels
    int resized_w = 0;
    int resized_h = 0;
    int input_w = 0;
    int input_h = 0;
};

bool letterbox(const cv::Mat& bgr,
               int input_w,
               int input_h,
               int padding_value,
               bool scaleup,
               bool center,
               ncnn::Mat& out,
               LetterBoxInfo& info);

void normalize_01_inplace(ncnn::Mat& in);

}  // namespace yolo26

