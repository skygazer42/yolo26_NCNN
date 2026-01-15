#pragma once

#include "ncnn/mat.h"

#include <vector>

namespace yolo26_seg {

void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis);
void interp(const ncnn::Mat& in, float scale, int out_w, int out_h, ncnn::Mat& out);
void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d);
void sigmoid(ncnn::Mat& bottom);
void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob);

}  // namespace yolo26_seg
