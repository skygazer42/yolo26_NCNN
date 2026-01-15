#pragma once

#include "yolo26.h"

#include <vector>

namespace ncnn {
class Mat;
}

namespace yolo26 {

void generate_proposals(const ncnn::Mat& reg, const ncnn::Mat& cls, int stride,
                        int input_w, int input_h, float prob_threshold,
                        std::vector<Yolo26Object>& objects);

void qsort_descent_inplace(std::vector<Yolo26Object>& objects);

void nms_sorted_bboxes(const std::vector<Yolo26Object>& objects, std::vector<int>& picked,
                       float nms_threshold);

}  // namespace yolo26
