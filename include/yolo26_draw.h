#pragma once

#include "yolo26.h"

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

const std::vector<std::string>& yolo26_coco_names();
const std::vector<cv::Scalar>& yolo26_coco_colors();
void yolo26_draw_objects(cv::Mat& bgr, const std::vector<Yolo26Object>& objects, float font_scale = 0.5f);
