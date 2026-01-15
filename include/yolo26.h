#pragma once

#include <opencv2/core/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ncnn {
class Net;
}

struct Yolo26Object {
    cv::Rect_<float> rect;
    int label = -1;
    float prob = 0.f;
};

struct Yolo26Config {
    int input_width = 640;
    int input_height = 640;
    int num_classes = 80;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    std::vector<int> strides = {8, 16, 32};
    bool use_gpu = false;
    std::string input_name = "data";
    std::vector<std::string> reg_blob_names = {"reg1", "reg2", "reg3"};
    std::vector<std::string> cls_blob_names = {"cls1", "cls2", "cls3"};
    std::string output_name = "output0";
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
