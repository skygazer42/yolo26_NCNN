#include "yolo26_preprocess.h"

#include <algorithm>
#include <cmath>

namespace yolo26 {

bool letterbox(const cv::Mat& bgr,
               int input_w,
               int input_h,
               int padding_value,
               bool scaleup,
               bool center,
               ncnn::Mat& out,
               LetterBoxInfo& info)
{
    if (bgr.empty() || input_w <= 0 || input_h <= 0)
        return false;

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    const float r0 = std::min(input_h / (float)img_h, input_w / (float)img_w);
    const float r = scaleup ? r0 : std::min(r0, 1.f);

    const int resized_w = std::max(1, (int)std::round(img_w * r));
    const int resized_h = std::max(1, (int)std::round(img_h * r));

    float dw = (float)input_w - resized_w;
    float dh = (float)input_h - resized_h;
    if (center)
    {
        dw /= 2.f;
        dh /= 2.f;
    }

    const int top = center ? (int)std::round(dh - 0.1f) : 0;
    const int bottom = (int)std::round(dh + 0.1f);
    const int left = center ? (int)std::round(dw - 0.1f) : 0;
    const int right = (int)std::round(dw + 0.1f);

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, resized_w, resized_h);

    ncnn::copy_make_border(resized, out,
                           top, bottom,
                           left, right,
                           ncnn::BORDER_CONSTANT, (float)padding_value);

    info.gain = r;
    info.pad_x = left;
    info.pad_y = top;
    info.resized_w = resized_w;
    info.resized_h = resized_h;
    info.input_w = input_w;
    info.input_h = input_h;

    return true;
}

void normalize_01_inplace(ncnn::Mat& in)
{
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);
}

}  // namespace yolo26

