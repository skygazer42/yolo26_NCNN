#include "yolo26.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#include "ncnn/net.h"
#include "ncnn/cpu.h"

#include "yolo26_postprocess.h"

Yolo26::Yolo26(const Yolo26Config& config)
    : config_(config), net_(std::make_shared<ncnn::Net>())
{
}

Yolo26::~Yolo26() = default;

bool Yolo26::load(const std::string& param_path, const std::string& bin_path)
{
    if (!net_)
        net_ = std::make_shared<ncnn::Net>();

#if NCNN_VULKAN
    net_->opt.use_vulkan_compute = config_.use_gpu;
#endif
    net_->opt.num_threads = ncnn::get_big_cpu_count();

    if (net_->load_param(param_path.c_str()) != 0)
        return false;
    if (net_->load_model(bin_path.c_str()) != 0)
        return false;

    return true;
}

bool Yolo26::detect(const cv::Mat& bgr, std::vector<Yolo26Object>& objects) const
{
    if (!net_ || bgr.empty())
        return false;

    if (config_.reg_blob_names.size() != config_.strides.size() ||
        config_.cls_blob_names.size() != config_.strides.size())
        return false;

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    float scale = std::min(config_.input_width / (float)img_w, config_.input_height / (float)img_h);
    int resized_w = std::max(1, (int)std::round(img_w * scale));
    int resized_h = std::max(1, (int)std::round(img_h * scale));

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, resized_w, resized_h);

    int pad_w = config_.input_width - resized_w;
    int pad_h = config_.input_height - resized_h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad,
                           pad_h / 2, pad_h - pad_h / 2,
                           pad_w / 2, pad_w - pad_w / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net_->create_extractor();
    int input_ret = ex.input(config_.input_name.c_str(), in_pad);
    if (input_ret != 0)
    {
        if (config_.input_name != "images")
            input_ret = ex.input("images", in_pad);
        if (input_ret != 0 && config_.input_name != "data")
            input_ret = ex.input("data", in_pad);
        if (input_ret != 0)
            return false;
    }

    bool raw_ok = true;
    std::vector<ncnn::Mat> regs(config_.reg_blob_names.size());
    std::vector<ncnn::Mat> clss(config_.cls_blob_names.size());
    for (size_t i = 0; i < config_.reg_blob_names.size(); i++)
    {
        if (ex.extract(config_.reg_blob_names[i].c_str(), regs[i]) != 0)
        {
            raw_ok = false;
            break;
        }
        if (ex.extract(config_.cls_blob_names[i].c_str(), clss[i]) != 0)
        {
            raw_ok = false;
            break;
        }
    }

    std::vector<Yolo26Object> proposals;
    if (raw_ok)
    {
        for (size_t i = 0; i < regs.size(); i++)
        {
            yolo26::generate_proposals(regs[i], clss[i], config_.strides[i],
                                       config_.input_width, config_.input_height,
                                       config_.conf_threshold, proposals);
        }

        yolo26::qsort_descent_inplace(proposals);

        std::vector<int> picked;
        yolo26::nms_sorted_bboxes(proposals, picked, config_.nms_threshold);

        objects.clear();
        objects.reserve(picked.size());
        for (size_t i = 0; i < picked.size(); i++)
        {
            objects.push_back(proposals[picked[i]]);
        }
    }
    else
    {
        ncnn::Mat out;
        int out_ret = ex.extract(config_.output_name.c_str(), out);
        if (out_ret != 0 && config_.output_name != "output")
            out_ret = ex.extract("output", out);
        if (out_ret != 0 && config_.output_name != "output0")
            out_ret = ex.extract("output0", out);
        if (out_ret != 0)
            return false;

        ncnn::Mat out_2d = out;
        if (out.dims == 3)
            out_2d = out.channel(0);

        if (out_2d.w < 6)
            return false;

        const int num_dets = out_2d.h;
        for (int i = 0; i < num_dets; i++)
        {
            const float* p = out_2d.row(i);
            float score = p[4];
            if (score < config_.conf_threshold)
                continue;

            Yolo26Object obj;
            obj.rect.x = p[0];
            obj.rect.y = p[1];
            obj.rect.width = p[2] - p[0];
            obj.rect.height = p[3] - p[1];
            obj.prob = score;
            obj.label = (int)p[5];
            proposals.push_back(obj);
        }

        objects.swap(proposals);
    }

    for (auto& obj : objects)
    {
        float x0 = (obj.rect.x - pad_w / 2.f) / scale;
        float y0 = (obj.rect.y - pad_h / 2.f) / scale;
        float x1 = (obj.rect.x + obj.rect.width - pad_w / 2.f) / scale;
        float y1 = (obj.rect.y + obj.rect.height - pad_h / 2.f) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
    }

    return true;
}
