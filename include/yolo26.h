#pragma once

#include <opencv2/core/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "yolo26_types.h"

namespace ncnn {
class Net;
}

struct Yolo26Object {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    int label = -1;
    float prob = 0.f;
};

struct Yolo26Config {
    int input_width = 640;
    int input_height = 640;
    int num_classes = 80;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_det = 300;
    int padding_value = 114;
    bool scaleup = true;
    bool center = true;
    Yolo26BoxFormat box_format = Yolo26BoxFormat::CXCYWH;
    Yolo26PostprocessType postprocess = Yolo26PostprocessType::Auto;
    bool topk_dedup = false;
    bool agnostic_nms = false;
    bool use_gpu = false;
    std::string input_name = "in0";
    std::string output_name = "out0";
};

class Yolo26 {
public:
    explicit Yolo26(const Yolo26Config& config = Yolo26Config());
    ~Yolo26();

    bool load(const std::string& param_path, const std::string& bin_path);
    bool detect(const cv::Mat& bgr, std::vector<Yolo26Object>& objects) const;

    const Yolo26Config& config() const { return config_; }

private:
    Yolo26Config config_;
    std::shared_ptr<ncnn::Net> net_;
};
